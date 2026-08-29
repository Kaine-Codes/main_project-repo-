import os
import glob
import re

def generate_summary(test_dir, base_chb_dir):
    edf_files = glob.glob(os.path.join(test_dir, '*.edf'))
    out_path = os.path.join(test_dir, 'test-summary.txt')
    
    with open(out_path, 'w') as out_f:
        for edf_path in edf_files:
            filename = os.path.basename(edf_path)
            # Extract the patient ID (e.g. 'chb01' from 'chb01_03.edf')
            match = re.match(r'(chb\d+)_', filename)
            if not match:
                print(f"Skipping {filename}: doesn't match expected pattern")
                continue
            patient_id = match.group(1)
            summary_path = os.path.join(base_chb_dir, patient_id, f'{patient_id}-summary.txt')
            
            if not os.path.exists(summary_path):
                print(f"Summary not found: {summary_path}")
                continue
                
            with open(summary_path, 'r') as f:
                content = f.read()
                
            # Split the summary into blocks for each EDF file
            blocks = re.split(r'(?=File Name:\s*\S+)', content)
            found = False
            for block in blocks:
                if f"File Name: {filename}" in block:
                    out_f.write(block.strip() + "\n\n")
                    found = True
                    break
            
            if not found:
                print(f"Could not find block for {filename} in {summary_path}")
                
    print(f"Done! Created {out_path}")

if __name__ == '__main__':
    generate_summary('CHB DATASET/test', 'CHB DATASET')
