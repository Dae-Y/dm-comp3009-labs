"""
Data Mining Lab02 - Data Preparation
Daehwan Yeo

Task 3: Q13 from Chapter 3 of Aggarwal
Demonstrate how the sample mean of an attribute approaches the population mean
as the sample size increases, using random sampling and z-score evaluation.

Dataset: Modified KDD Cup 1999 (.arff format)
"""

import pandas as pd
import numpy as np
from scipy.io import arff
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load the Data
# -----------------------------
# File assumed to be in the same directory
file_name = "kddcup99.arff"
data = arff.loadarff(file_name)
df = pd.DataFrame(data[0])

# Convert byte string columns to normal strings
df = df.map(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

# -----------------------------
# Step 2: Compute Population Stats
# -----------------------------
df["count"] = pd.to_numeric(df["count"])
sample_data = df["count"][:10000]  # first 10,000 rows

mu = sample_data.mean()
sigma = sample_data.std()

print(f"Population mean (μ): {mu:.2f}")
print(f"Population std deviation (σ): {sigma:.2f}")

# -----------------------------
# Step 3: Random Sampling & z-scores
# -----------------------------
sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
average_zn = []

for n in sample_sizes:
    z_values = []
    for _ in range(10):  # repeat sampling 10 times
        sample = sample_data.sample(n=n, replace=False,
                                    random_state=np.random.randint(10000))
        en = sample.mean()
        zn = abs(en - mu) / sigma
        z_values.append(zn)
    average_zn.append(np.mean(z_values))

# -----------------------------
# Step 4: Plot the Results
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, average_zn, marker='o')
plt.xscale("log")
plt.xlabel("Sample Size (n)")
plt.ylabel("Average z-score (ẑₙ)")
plt.title("Average z-score vs Sample Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("ex3_AvgZscoreSampleSize.png")
plt.close()
print("Saved plot: ex3_AvgZscoreSampleSize.png")

# -----------------------------
# Step 5: Save Reflection
# -----------------------------
reflection = """\
Reflection on Sample Mean Convergence

Trend as sample size increases:
As n increases, the average z-score (ẑₙ) decreases steadily.
Small samples (n=10, 20) have large deviations, while larger samples
approach the population mean more closely, with z-scores trending toward zero.

Why z-score is useful:
The z-score normalizes deviation in terms of standard deviations,
making it independent of the raw scale of the 'count' attribute.
This allows meaningful comparison of deviations across sample sizes.
"""

with open("ex3_result.txt", "w", encoding="utf-8") as f:
    f.write(reflection)

print("Saved reflection: ex3_result.txt")
