import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 1. Load heart.csv
df = pd.read_csv("heart.csv")
print("Shape:", df.shape)
print(df.head())

cont_cols = ["age", "chol", "trestbps", "thalach"]
print("\nSummary statistics for continuous columns:")
print(df[cont_cols].describe())

# 2. Histograms (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, cont_cols):
    ax.hist(df[col].dropna(), bins=20, edgecolor="black")
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("PW3/heart_distributions.png", dpi=200)
plt.close()

# 3. Normality checks (Shapiro-Wilk + Q-Q plot)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
print("\nNormality checks (Shapiro-Wilk test):")
for ax, col in zip(axes.flat, cont_cols):
    series = df[col].dropna()
    stat, p_val = stats.shapiro(series)
    verdict = "Approximately normal" if p_val > 0.05 else "Deviates from normality"
    print(f"  {col:10s}: W = {stat:.4f}, p = {p_val:.4e} -> {verdict}")
    stats.probplot(series, dist="norm", plot=ax)
    ax.set_title(f"Q-Q Plot: {col}")
plt.tight_layout()
plt.savefig("PW3/heart_qqplots.png", dpi=200)
plt.close()

# 4. Compare thalach between target=0 (no disease) and target=1 (disease)
g0 = df.loc[df["target"] == 0, "thalach"].dropna()
g1 = df.loc[df["target"] == 1, "thalach"].dropna()

mean0, sem0 = g0.mean(), stats.sem(g0)
mean1, sem1 = g1.mean(), stats.sem(g1)
ci0 = stats.t.interval(0.95, df=len(g0) - 1, loc=mean0, scale=sem0)
ci1 = stats.t.interval(0.95, df=len(g1) - 1, loc=mean1, scale=sem1)

u_stat, p_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
t_stat, p_welch = stats.ttest_ind(g0, g1, equal_var=False)

print(f"\nTarget=0 (no disease): n={len(g0)}, mean={mean0:.2f}, SEM={sem0:.2f}, 95% CI=({ci0[0]:.2f}, {ci0[1]:.2f})")
print(f"Target=1 (disease):    n={len(g1)}, mean={mean1:.2f}, SEM={sem1:.2f}, 95% CI=({ci1[0]:.2f}, {ci1[1]:.2f})")
print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4e} | Welch's t: t={t_stat:.4f}, p={p_welch:.4e}")

plt.figure(figsize=(6, 4.5))
means = [mean0, mean1]
cis = [ci0[1] - mean0, ci1[1] - mean1]
plt.bar(["No Disease (0)", "Heart Disease (1)"], means, yerr=cis, capsize=6)
plt.ylabel("Maximum Heart Rate (thalach)")
plt.title("Mean thalach by Target (with 95% CI)")
plt.tight_layout()
plt.savefig("PW3/thalach_by_target.png", dpi=200)
plt.close()

# 5. Relationship between age and thalach
r_pear, p_pear = stats.pearsonr(df["age"], df["thalach"])
r_spear, p_spear = stats.spearmanr(df["age"], df["thalach"])
print(f"\nage vs thalach: Pearson r={r_pear:.4f} (p={p_pear:.4e}), Spearman rho={r_spear:.4f} (p={p_spear:.4e})")

plt.figure(figsize=(6.5, 4.5))
plt.scatter(df["age"], df["thalach"], alpha=0.6)
m, b = np.polyfit(df["age"], df["thalach"], 1)
x_line = np.linspace(df["age"].min(), df["age"].max(), 100)
plt.plot(x_line, m * x_line + b)
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate (thalach)")
plt.title("Age vs. Maximum Heart Rate (thalach)")
plt.tight_layout()
plt.savefig("PW3/age_vs_thalach.png", dpi=200)
plt.close()

# Bonus: Target balance & Shannon entropy (normalized)
props = df["target"].value_counts(normalize=True)
entropy = -np.sum(props * np.log2(props))
print("\nTarget proportions:\n", props)
print(f"Binary entropy (max=1.0 for 50/50 split): {entropy:.4f}")