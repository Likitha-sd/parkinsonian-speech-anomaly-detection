from pathlib import Path
import pandas as pd


DATA_DIR = Path(
    r"C:\Users\Likithasri\Downloads\parkinson+speech+dataset+with+multiple+types+of+sound+recordings"
)

train_path = DATA_DIR / "train_data.txt"
test_path = DATA_DIR / "test_data.txt"


# Load data
train = pd.read_csv(train_path, header=None)
test = pd.read_csv(test_path, header=None)


print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print("Train shape:", train.shape)
print("Test shape:", test.shape)


print("\n" + "=" * 60)
print("FIRST 5 TRAINING ROWS")
print("=" * 60)

print(train.head())


print("\n" + "=" * 60)
print("FIRST 5 TEST ROWS")
print("=" * 60)

print(test.head())


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print("Train missing values:")
print(train.isnull().sum())

print("\nTest missing values:")
print(test.isnull().sum())


print("\n" + "=" * 60)
print("TRAIN DATA TYPES")
print("=" * 60)

print(train.dtypes)


print("\n" + "=" * 60)
print("UNIQUE VALUES PER COLUMN")
print("=" * 60)

for column in train.columns:
    print(
        f"Column {column}: "
        f"{train[column].nunique()} unique values"
    )