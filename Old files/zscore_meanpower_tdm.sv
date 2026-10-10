// zscore_meanpower_tdm.sv
// Blocks B+C: Fused 6-Stage Z-Score Normalization + Mean Power Pipeline (TDM)
//
// Processes N_CH electrodes through a single shared pipeline via TDM.
// On each batch_valid pulse, channels 0..N_CH-1 are fed sequentially.
//
// 6-Stage Pipeline (1 channel/cycle throughput, 6-cycle latency):
//   Stage 1: Mean subtraction     diff = x_raw - mu_tau[ch]
//   Stage 2: Scaling              prod = diff * (1/sigma_tau[ch])
//   Stage 3: Round + Clamp +/-6   z    = clamp(prod >> shift, +/-6.0)
//   Stage 4: Square               z_sq = z * z
//   Stage 5: Accumulate           acc[ch] += z_sq
//   Stage 6: Store                if WIN samples done: mean_power[ch] = acc >> shift
//
// Input:  Q8.16 (24-bit signed)
// Output: Q4.12 (16-bit signed) mean power per channel
//
// Lint-clean for Verilator. Questa/Vivado compatible.
`timescale 1ns / 1ps

module zscore_meanpower_tdm #(
    parameter N_CH   = 2,
    parameter ADC_W  = 24,          // Raw ADC width (Q8.16)
    parameter FRAC_I = 16,          // Input fractional bits
    parameter OUT_W  = 16,          // Output width (Q4.12)
    parameter FRAC_O = 12,          // Output fractional bits
    parameter WIN    = 256          // Samples per 1-second window
)(
    input  logic                       clk,
    input  logic                       rst_n,

    // --- Batch input (all channels sampled simultaneously by ADC) ---
    input  logic signed [ADC_W-1:0]    x_batch     [0:N_CH-1],
    input  logic                       batch_valid,

    // --- Calibration register file (loaded once per patient) ---
    input  logic signed [ADC_W-1:0]    mu_reg      [0:N_CH-1],
    input  logic signed [OUT_W-1:0]    inv_sig_reg [0:N_CH-1],

    // --- Mean power output (one per channel, Q4.12) ---
    output logic signed [OUT_W-1:0]    mean_power  [0:N_CH-1],
    output logic                       window_valid    // Pulse when all mean_power[] updated
);

    // ---------------------------------------------------------------
    // Derived parameters
    // ---------------------------------------------------------------
    localparam CH_W      = $clog2(N_CH > 1 ? N_CH : 2);
    localparam PROD_W    = ADC_W + OUT_W;               // 40-bit product
    localparam FRAC_PROD = FRAC_I + FRAC_O;             // 28 fractional bits in product
    localparam SHIFT_Z   = FRAC_PROD - FRAC_O;          // 16 bits to discard for z
    localparam ROUND_Z   = SHIFT_Z - 1;                 // Bit 15 for round-to-nearest

    // Z-score clamp thresholds in Q4.12
    localparam signed [OUT_W-1:0] CLAMP_POS =  16'sd24576;   // +6.0 * 4096
    localparam signed [OUT_W-1:0] CLAMP_NEG = -16'sd24576;   // -6.0 * 4096
    localparam signed [PROD_W-1:0] CLAMP_POS_EXT = PROD_W'(signed'(CLAMP_POS));
    localparam signed [PROD_W-1:0] CLAMP_NEG_EXT = PROD_W'(signed'(CLAMP_NEG));

    // z^2 product: Q4.12 * Q4.12 = Q8.24 (32-bit)
    localparam ZSQ_W    = OUT_W * 2;                    // 32
    localparam FRAC_ZSQ = FRAC_O * 2;                   // 24

    // Accumulator: 40-bit to safely hold sum of 256 * z^2
    // Max z^2 = 36.0 in Q8.24 = 603,979,776. Sum of 256 = ~154B → 38 bits needed.
    localparam ACC_W = 40;

    // Mean power shift: acc >> 8 (divide by 256) then Q8.24 → Q4.12 (>> 12)
    // Total: >> 20
    localparam SHIFT_MP = 8 + (FRAC_ZSQ - FRAC_O);     // 8 + 12 = 20
    localparam ROUND_MP = SHIFT_MP - 1;                 // 19

    // ---------------------------------------------------------------
    // TDM Feed FSM: sequences channels into the pipeline
    // ---------------------------------------------------------------
    typedef enum logic [1:0] { S_IDLE, S_FEED } state_t;
    state_t state;

    logic [CH_W-1:0]           feed_idx;
    logic signed [ADC_W-1:0]   x_latched [0:N_CH-1];

    // Pipeline input signals
    logic signed [ADC_W-1:0]   pipe_x;
    logic                      pipe_valid;
    logic signed [ADC_W-1:0]   pipe_mu;
    logic signed [OUT_W-1:0]   pipe_inv;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            feed_idx   <= '0;
            pipe_valid <= 1'b0;
            for (int i = 0; i < N_CH; i++) x_latched[i] <= '0;
        end else begin
            case (state)
                S_IDLE: begin
                    pipe_valid <= 1'b0;
                    if (batch_valid) begin
                        for (int i = 0; i < N_CH; i++)
                            x_latched[i] <= x_batch[i];
                        state      <= S_FEED;
                        feed_idx   <= '0;
                        pipe_valid <= 1'b1;
                    end
                end

                S_FEED: begin
                    if (feed_idx == CH_W'(N_CH - 1)) begin
                        state      <= S_IDLE;
                        pipe_valid <= 1'b0;
                    end else begin
                        feed_idx   <= feed_idx + 1;
                        pipe_valid <= 1'b1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // Mux: select channel data + constants based on feed_idx
    assign pipe_x   = x_latched[feed_idx];
    assign pipe_mu  = mu_reg[feed_idx];
    assign pipe_inv = inv_sig_reg[feed_idx];

    // ---------------------------------------------------------------
    // Channel ID delay line (tracks channel through 6 stages)
    // ch_pipe[0] = channel entering stage 1
    // ch_pipe[k] = channel exiting stage k
    // ---------------------------------------------------------------
    logic [CH_W-1:0] ch_pipe [0:5];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < 6; i++) ch_pipe[i] <= '0;
        end else begin
            ch_pipe[0] <= feed_idx;
            for (int i = 1; i < 6; i++)
                ch_pipe[i] <= ch_pipe[i-1];
        end
    end

    // ---------------------------------------------------------------
    // Stage 1: Mean subtraction  (diff = x_raw - mu_tau)
    // ---------------------------------------------------------------
    logic signed [ADC_W-1:0] diff_s1;
    logic signed [OUT_W-1:0] inv_sigma_s1;
    logic                    valid_s1;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            diff_s1      <= '0;
            inv_sigma_s1 <= '0;
            valid_s1     <= 1'b0;
        end else begin
            valid_s1 <= pipe_valid;
            if (pipe_valid) begin
                diff_s1      <= pipe_x - pipe_mu;
                inv_sigma_s1 <= pipe_inv;
            end
        end
    end

    // ---------------------------------------------------------------
    // Stage 2: Scaling  (prod = diff * inv_sigma)   [DSP48]
    // ---------------------------------------------------------------
    logic signed [PROD_W-1:0] prod_s2;
    logic                     valid_s2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            prod_s2  <= '0;
            valid_s2 <= 1'b0;
        end else begin
            valid_s2 <= valid_s1;
            if (valid_s1)
                prod_s2 <= diff_s1 * inv_sigma_s1;
        end
    end

    // ---------------------------------------------------------------
    // Stage 3: Round + shift + saturation clamp ±6σ
    // ---------------------------------------------------------------
    logic signed [OUT_W-1:0] z_s3;
    logic                    valid_s3;

    always_ff @(posedge clk or negedge rst_n) begin
        logic signed [PROD_W-1:0] rounded;
        logic signed [PROD_W-1:0] shifted;

        if (!rst_n) begin
            z_s3     <= '0;
            valid_s3 <= 1'b0;
        end else begin
            valid_s3 <= valid_s2;
            if (valid_s2) begin
                rounded = prod_s2 + (PROD_W'(1) <<< ROUND_Z);
                shifted = rounded >>> SHIFT_Z;

                if (shifted > CLAMP_POS_EXT)
                    z_s3 <= CLAMP_POS;
                else if (shifted < CLAMP_NEG_EXT)
                    z_s3 <= CLAMP_NEG;
                else
                    z_s3 <= shifted[OUT_W-1:0];
            end
        end
    end

    // ---------------------------------------------------------------
    // Stage 4: Square  (z_sq = z * z)   [DSP48]
    // ---------------------------------------------------------------
    logic signed [ZSQ_W-1:0] zsq_s4;
    logic                    valid_s4;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            zsq_s4   <= '0;
            valid_s4 <= 1'b0;
        end else begin
            valid_s4 <= valid_s3;
            if (valid_s3)
                zsq_s4 <= z_s3 * z_s3;
        end
    end

    // ---------------------------------------------------------------
    // Stages 5+6: Accumulate + Window Store (fused to avoid multi-driver)
    //
    // Stage 5: acc[ch] += z_sq,  samp_cnt[ch] += 1
    // Stage 6: if samp_cnt == WIN → store mean_power, reset acc & cnt
    //
    // These share the acc/samp_cnt register file, so they are combined
    // into one always_ff block. The window check uses the UPDATED count.
    // ---------------------------------------------------------------
    logic signed [ACC_W-1:0] acc       [0:N_CH-1];
    logic [8:0]              samp_cnt  [0:N_CH-1];   // 0..256 → 9 bits
    logic [CH_W-1:0]         done_cnt;

    // Channel at accumulation = ch_pipe[3]
    // valid_s4 is 4 delays from pipe_valid; ch_pipe[3] is 4 delays from feed_idx
    // (ch_pipe[0]=1 delay, [1]=2, [2]=3, [3]=4). They are synchronized.
    wire [CH_W-1:0] ch_at_acc = ch_pipe[3];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            window_valid <= 1'b0;
            done_cnt     <= '0;
            for (int i = 0; i < N_CH; i++) begin
                acc[i]         <= '0;
                samp_cnt[i]    <= '0;
                mean_power[i]  <= '0;
            end
        end else begin
            window_valid <= 1'b0;

            if (valid_s4) begin
                // Compute new accumulator and count
                automatic logic signed [ACC_W-1:0] new_acc;
                automatic logic [8:0]              new_cnt;
                new_acc = acc[ch_at_acc] + ACC_W'(signed'(zsq_s4));
                new_cnt = samp_cnt[ch_at_acc] + 9'd1;

                if (new_cnt == WIN[8:0]) begin
                    // ---- Window complete: store mean power, reset ----
                    // Round-to-nearest then shift
                    automatic logic signed [ACC_W-1:0] rounded_acc;
                    rounded_acc = new_acc + (ACC_W'(1) << ROUND_MP);
                    mean_power[ch_at_acc] <= OUT_W'(rounded_acc >>> SHIFT_MP);

                    // Reset accumulator and counter for next window
                    acc[ch_at_acc]      <= '0;
                    samp_cnt[ch_at_acc] <= '0;

                    // Track how many channels have completed this window
                    if (done_cnt == CH_W'(N_CH - 1)) begin
                        window_valid <= 1'b1;
                        done_cnt     <= '0;
                    end else begin
                        done_cnt <= done_cnt + 1;
                    end
                end else begin
                    // ---- Normal accumulation ----
                    acc[ch_at_acc]      <= new_acc;
                    samp_cnt[ch_at_acc] <= new_cnt;
                end
            end
        end
    end

endmodule
