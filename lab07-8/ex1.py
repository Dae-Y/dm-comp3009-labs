# ex1.py
# Author: Daehwan Yeo
# Description: Generate synthetic data, run multiple clustering algorithms,
#              save visualisations and metrics (including per-cluster SSQ) to ./ex1

# --- 1. Standard Imports ---
import os
import sys
import numpy as np
import pandas as pd

# Use a non-interactive backend so the script never blocks on plt.show()
import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

# --- 2. Helper Functions ---

def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())

def plot3d(X, y, title='3D Scatter Plot', output_dir='.', show=False):
    """
    Save a 3D scatter plot for the given data and labels.
    Noise label '-1' (e.g., from DBSCAN) is rendered as 'Noise'.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = pd.unique(y)
    for lab in unique_labels:
        mask = (y == lab)
        label_txt = f'Cluster {lab}' if lab != -1 else 'Noise'
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2], label=label_txt, s=14)

    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend(loc='best')

    filename = _safe_name(title.lower().replace(' ', '_')) + '.png'
    filepath = os.path.join(output_dir, filename)
    fig.tight_layout()
    plt.savefig(filepath, dpi=140)
    if show:
        plt.show()
    plt.close(fig)

def plot_dendrogram(model, output_dir='.', **kwargs):
    """
    Create and save a dendrogram from a fitted AgglomerativeClustering model
    that was trained with distance_threshold=0 and compute_distances=True.
    """
    if not hasattr(model, "distances_"):
        raise ValueError("model.distances_ is missing. Fit with compute_distances=True and distance_threshold=0.")

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    fig = plt.figure(figsize=(12, 8))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.ylabel("Distance")
    filepath = os.path.join(output_dir, 'dendrogram.png')
    fig.tight_layout()
    plt.savefig(filepath, dpi=140)
    plt.close(fig)

def calc_ssq(X, labels):
    """
    Compute per-cluster SSQ and total SSQ.
    Skips label -1 (noise) if present (e.g., DBSCAN).
    Returns: (total_ssq: float, per_cluster_ssq: dict[label -> float])
    """
    per = {}
    for label in np.unique(labels):
        if label == -1:
            continue  # ignore noise
        pts = X[labels == label]
        if pts.size == 0:
            per[int(label)] = 0.0
            continue
        centroid = np.mean(pts, axis=0)
        diffs = pts - centroid
        per[int(label)] = float(np.sum(diffs ** 2))
    total = float(np.sum(list(per.values()))) if per else 0.0
    return total, dict(sorted(per.items(), key=lambda kv: kv[0]))

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

# --- 3. Main Execution Block ---

def main():
    """
    Orchestrates data generation, clustering, evaluation, and saving of outputs.
    """
    output_dir = 'ex1'
    ensure_dir(output_dir)

    log_file_path = os.path.join(output_dir, 'result.txt')
    original_stdout = sys.stdout

    with open(log_file_path, 'w', encoding='utf-8') as f:
        sys.stdout = f

        # --- Generate synthetic data ---
        X, y_true = make_classification(
            n_samples=1000,
            n_features=3,
            n_informative=3,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=3,
            class_sep=3.0,
            random_state=123
        )

        print("--- Visualising the Original Data ---")
        plot3d(X, y_true, title='Original Data with True Labels', output_dir=output_dir, show=False)

        results_rows = []  # for summary.csv
        print("\n--- Agglomerative Clustering (full tree for dendrogram) ---")
        model_ac_full = AgglomerativeClustering(distance_threshold=0, n_clusters=None, compute_distances=True)
        model_ac_full.fit(X)
        plot_dendrogram(model_ac_full, truncate_mode='level', p=5, output_dir=output_dir)
        print("Saved dendrogram as ex1/dendrogram.png")

        # Decide number of clusters from dendrogram; here we follow the lab and use 3.
        print("\n--- Agglomerative Clustering (n_clusters=3) ---")
        model_ac = AgglomerativeClustering(n_clusters=3)
        yhat_ac = model_ac.fit_predict(X)
        total_ac, per_ac = calc_ssq(X, yhat_ac)
        print("Per-cluster SSQ (Agglomerative):")
        for c, v in per_ac.items():
            print(f"  Cluster {c}: {v:,.2f}")
        print(f"Total SSQ (Agglomerative): {total_ac:,.2f}")
        plot3d(X, yhat_ac, title='Agglomerative Clustering Results', output_dir=output_dir, show=False)
        results_rows.append({
            "method": "Agglomerative",
            "n_clusters": len(per_ac),
            "noise_points": 0,
            "ssq_total": total_ac
        })

        # --- K-Means ---
        print("\n--- K-Means (k=3) ---")
        model_km = KMeans(n_clusters=3, random_state=123, n_init=10)
        yhat_km = model_km.fit_predict(X)
        total_km, per_km = calc_ssq(X, yhat_km)
        print("Per-cluster SSQ (K-Means):")
        for c, v in per_km.items():
            print(f"  Cluster {c}: {v:,.2f}")
        print(f"Total SSQ (K-Means): {total_km:,.2f}")
        plot3d(X, yhat_km, title='K-Means Clustering Results', output_dir=output_dir, show=False)
        results_rows.append({
            "method": "K-Means",
            "n_clusters": len(per_km),
            "noise_points": 0,
            "ssq_total": total_km
        })

        # --- Gaussian Mixture ---
        print("\n--- Gaussian Mixture Model (n_components=3) ---")
        model_gm = GaussianMixture(n_components=3, random_state=123)
        yhat_gm = model_gm.fit_predict(X)
        total_gm, per_gm = calc_ssq(X, yhat_gm)
        print("Per-cluster SSQ (Gaussian Mixture):")
        for c, v in per_gm.items():
            print(f"  Cluster {c}: {v:,.2f}")
        print(f"Total SSQ (Gaussian Mixture): {total_gm:,.2f}")
        plot3d(X, yhat_gm, title='Gaussian Mixture Results', output_dir=output_dir, show=False)
        results_rows.append({
            "method": "Gaussian Mixture",
            "n_clusters": len(per_gm),
            "noise_points": 0,
            "ssq_total": total_gm
        })

        # --- DBSCAN ---
        print("\n--- DBSCAN ---")
        # eps was 1.5 in your draft; with class_sep=3 this is reasonable.
        model_db = DBSCAN(eps=1.5, min_samples=5)
        yhat_db = model_db.fit_predict(X)

        unique_labels, counts = np.unique(yhat_db, return_counts=True)
        n_clusters_found = int(np.sum(unique_labels != -1))
        noise_points = int(counts[unique_labels == -1][0]) if (-1 in unique_labels) else 0

        print(f"DBSCAN found {n_clusters_found} clusters.")
        print(f"Number of noise points (label -1): {noise_points}")

        total_db, per_db = calc_ssq(X, yhat_db)
        print("Per-cluster SSQ (DBSCAN) [noise excluded]:")
        for c, v in per_db.items():
            print(f"  Cluster {c}: {v:,.2f}")
        print(f"Total SSQ (DBSCAN): {total_db:,.2f}")
        plot3d(X, yhat_db, title='DBSCAN Clustering Results', output_dir=output_dir, show=False)
        results_rows.append({
            "method": "DBSCAN",
            "n_clusters": len(per_db),
            "noise_points": noise_points,
            "ssq_total": total_db
        })

        # --- Final comparison ---
        print("\n--- Comparison of Clustering Methods by Total SSQ (lower is better) ---")
        summary_df = pd.DataFrame(results_rows)
        for _, row in summary_df.iterrows():
            print(f"{row['method']:>16} | clusters={row['n_clusters']:>2} | noise={row['noise_points']:>3} | SSQ={row['ssq_total']:,.2f}")

        best_idx = summary_df['ssq_total'].idxmin()
        best_method = summary_df.loc[best_idx, 'method']
        print(f"\nBased on Total SSQ, the best clustering method is: '{best_method}'.")

        # Save CSV summary
        csv_path = os.path.join(output_dir, "summary.csv")
        summary_df.to_csv(csv_path, index=False)
        print(f"\nSaved summary to {csv_path}")

    # Restore stdout and notify
    sys.stdout = original_stdout
    print(f"Analysis complete. Results saved to '{log_file_path}'.")
    print(f"Plots and CSV saved in the '{output_dir}/' directory.")

if __name__ == '__main__':
    main()
