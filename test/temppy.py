import pandas as pd

file = r"S:\projects\Main Project\test\output\chb08_05_features.xlsx"

df = pd.read_excel(
    file,
    sheet_name="Features_Long"
)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nFIRST 10 ROWS:")
print(df.head(10).to_string())

print("\nUNIQUE VALUES OF POSSIBLE LABEL/STAGE COLUMNS:")

for col in df.columns:
    name = str(col).lower()

    if any(x in name for x in [
        "stage",
        "label",
        "class",
        "state",
        "seizure"
    ]):
        print("\n", col)
        print(df[col].unique()[:30])