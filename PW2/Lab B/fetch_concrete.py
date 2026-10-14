import os
from ucimlrepo import fetch_ucirepo

# 1. Fetch the dataset using its UCI ID (165)[cite: 1, 2]
concrete = fetch_ucirepo(id=165)
X = concrete.data.features
y = concrete.data.targets

# Print metadata and variables table to inspect[cite: 1, 2]
print(concrete.metadata)
print(concrete.variables)

# 2. Rename columns by position
X = X.copy()
X.columns = [
    "cement",
    "slag",
    "ash",
    "coarse_agg",
    "fine_agg",
    "water",
    "superplastic",
    "age",
]

# Convert target dataframe to a 1-D Series named "strength"
y = y.iloc[:, 0].rename("strength")

# 3. Save to data/concrete.csv[cite: 2]
os.makedirs("data", exist_ok=True)
X.assign(strength=y).to_csv("data/concrete.csv", index=False)