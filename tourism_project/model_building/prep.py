import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

# Load the data from the repo data folder
df = pd.read_csv(DATA_PATH)

# Drop columns we do not need (row index and customer id)
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

# Fix the typo in Gender
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Split features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# 80/20 split, keeping the buyer ratio the same in both
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally. The workflow passes these files to the next job as an artifact.
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train:", Xtrain.shape, "| Test:", Xtest.shape)
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
