import pandas as pd

#  Load data
DATA_PATH = "/home/runner/work/tourism-package-prediction/tourism-package-prediction/tourism_project/data/tourism.csv"
df_original = pd.read_csv(DATA_PATH)
df = df_original.copy()
print(f'Shape: {df.shape}')

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]



# Check that every expected column is there
missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
assert not missing, f"Missing columns: {missing}"
print("All expected columns found.")

# Short summary
print("Rows:", df.shape[0], "| Columns:", df.shape[1])
print("Buyers (ProdTaken=1): {:.1%}".format(df["ProdTaken"].mean()))
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("Dataset registered.")
