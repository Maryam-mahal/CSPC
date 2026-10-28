import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("PW3/chemicals_cancer.csv")
print("chemicals_cancer shape:", df.shape)
print(df.head())

# Task 6: Naive correlations with malignancy
for col in ["benzene", "cadmium", "pollution_index", "age"]:
    r_p, p_p = stats.pearsonr(df[col], df["malignancy"])
    r_s, p_s = stats.spearmanr(df[col], df["malignancy"])
    print(f"[Naive] {col:16s} -> Pearson r={r_p:.4f} (p={p_p:.2e}), Spearman rho={r_s:.4f} (p={p_s:.2e})")

# Task 7: Restrict pollution_index to (40, 60) to control for confounding
sub = df[(df["pollution_index"] > 40) & (df["pollution_index"] < 60)]
print(f"\nRestricted to 40 < pollution_index < 60 (n={len(sub)}):")
for col in ["benzene", "cadmium", "age"]:
    r_p, p_p = stats.pearsonr(sub[col], sub["malignancy"])
    r_s, p_s = stats.spearmanr(sub[col], sub["malignancy"])
    print(f"[Controlled] {col:13s} -> Pearson r={r_p:.4f} (p={p_p:.2e}), Spearman rho={r_s:.4f} (p={p_s:.2e})")

# Plot before vs. after controlling for pollution_index
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].scatter(df["cadmium"], df["malignancy"], alpha=0.35, label="Cadmium (all)")
axes[0].scatter(df["benzene"], df["malignancy"], alpha=0.35, label="Benzene (all)")
axes[0].set_title("Naive: Full Dataset (n=1000)")
axes[0].set_xlabel("Chemical level")
axes[0].set_ylabel("Malignancy score")
axes[0].legend()

axes[1].scatter(sub["cadmium"], sub["malignancy"], alpha=0.6, label="Cadmium (poll. 40-60)")
axes[1].scatter(sub["benzene"], sub["malignancy"], alpha=0.6, label="Benzene (poll. 40-60)")
axes[1].set_title("Controlled: Pollution Index in (40, 60)")
axes[1].set_xlabel("Chemical level")
axes[1].set_ylabel("Malignancy score")
axes[1].legend()

plt.tight_layout()
plt.savefig("PW3/chemicals_confounding.png", dpi=200)
plt.close()