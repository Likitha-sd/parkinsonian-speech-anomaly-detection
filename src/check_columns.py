from pathlib import Path
import pandas as pd


DATA_DIR = Path(
    r"C:\Users\Likithasri\Downloads\parkinson+speech+dataset+with+multiple+types+of+sound+recordings"
)

train = pd.read_csv(DATA_DIR / "train_data.txt", header=None)
test = pd.read_csv(DATA_DIR / "test_data.txt", header=None)

print("TRAIN")
print("=" * 60)

for col in train.columns:
    print(
        f"Column {col:2d} | "
        f"min={train[col].min():12.6f} | "
        f"max={train[col].max():12.6f} | "
        f"unique={train[col].nunique():4d}"
    )


print("\nTEST")
print("=" * 60)

for col in test.columns:
    print(
        f"Column {col:2d} | "
        f"min={test[col].min():12.6f} | "
        f"max={test[col].max():12.6f} | "
        f"unique={test[col].nunique():4d}"
    )