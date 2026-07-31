# Project 2 Deep-Dive Notes — Music Taste Mapper

Concepts to understand deeply enough to teach. Organized from foundational to sophisticated.

---

## 1. Supervised vs Unsupervised Learning — The Core Distinction

**Supervised learning** (Project 1 neural net): You provide labeled examples. The model learns a mapping from input features to known outputs. Training is driven by a loss function that measures prediction error against ground truth labels.

**Unsupervised learning** (Project 2 K-means): No labels. The model finds structure that exists in the data itself — patterns, groupings, or compressed representations — without being told what to look for. There is no "correct answer" to compare against.

This distinction changes everything:
- No labels → no loss function driven by ground truth → no backpropagation
- K-means minimizes a geometric objective (inertia) through direct centroid updates, not gradients
- "Accuracy" on unsupervised output is a consistency check, not a correctness check

The labels K-means produces are invented by the algorithm. They're meaningful only if the discovered clusters correspond to real structure in the data — which you validate qualitatively (do the cluster members make sense?) and quantitatively (silhouette score).

---

## 2. K-Means — Full Theory

### What it's doing

K-means partitions n data points into k clusters by minimizing **inertia**: the sum of squared Euclidean distances from each point to its assigned centroid.

Formally: minimize Σ Σ ||x - μ_i||²

where the outer sum is over clusters, the inner sum is over points in that cluster, x is a data point, and μ_i is the centroid of cluster i.

### The algorithm step by step

1. Initialize k centroids (randomly or via k-means++)
2. **Assignment step:** assign each point to its nearest centroid
3. **Update step:** recompute each centroid as the mean of all points assigned to it
4. Repeat steps 2–3 until assignments stop changing (convergence)

This is guaranteed to converge because:
- Each assignment step can only decrease or maintain inertia (points move to nearer centroids)
- Each update step can only decrease or maintain inertia (the mean minimizes sum of squared distances)
- There are finitely many possible assignments

### Why it can get stuck in local optima

K-means finds a local minimum of inertia, not the global minimum. The solution depends on where the centroids start. Two different random initializations can converge to completely different cluster assignments.

This is why `n_init=10` is important — sklearn runs the whole algorithm 10 times from different starting points and returns the best result (lowest inertia). Without this, you might consistently get bad solutions on some datasets.

### Why inertia always decreases with more k

At k=1, one centroid must cover all points — inertia is high. At k=n (every point is its own cluster), inertia is zero. Adding more clusters always gives points somewhere closer to go. This is why you can't pick k by minimizing inertia alone — you'd always pick k=n. The elbow method looks for diminishing returns.

### Why you square the distances

Distance itself is never negative (it's a magnitude). The squaring is a deliberate choice, not a safeguard:

1. **Penalizes outliers more:** a point distance 4 from its centroid contributes 16 to inertia, not just 4. K-means strongly avoids outlier points being far from centroids.
2. **Makes math clean:** the point that minimizes sum of squared distances to a set of points is exactly the mean. This is why the update step is a simple average. If you used absolute distances instead (K-medians), you'd need the median, which is slower to compute.

### Why scaling is mandatory for K-means

K-means uses Euclidean distance. If loudness ranges from -60 to 0 dB and danceability ranges from 0 to 1, a one-unit difference in loudness dominates a one-unit difference in danceability purely because of scale. The algorithm would effectively ignore low-range features. StandardScaler brings all features to mean=0, std=1 so each feature contributes equally to distance.

---

## 3. K-Means++ Initialization

### The problem with random initialization

Pure random initialization places k centroids randomly among data points. If two centroids start very close together, they'll converge to nearly identical clusters, wasting capacity. Worse, if all centroids start in one dense region, sparse regions of the data may never get a centroid.

### How k-means++ fixes this

K-means++ (Arthur & Vassilvitskii, 2007) uses probabilistic distance-weighted initialization:

1. Pick the first centroid uniformly at random from the data points
2. For each remaining data point, compute its distance to the nearest already-chosen centroid
3. Pick the next centroid with probability proportional to that squared distance — points far from existing centroids are more likely to be chosen
4. Repeat steps 2–3 until k centroids are chosen
5. Then run standard K-means from these starting points

**The intuition:** farther points are more likely to be chosen as new centroids, so the initial centroids tend to be spread out across the data rather than clustered together. This gives K-means a much better starting position, leading to faster convergence and better final solutions.

**In sklearn:** `init='k-means++'` is actually the default in recent versions. Setting it explicitly is good practice for clarity and version stability.

**The guarantee:** K-means++ initialization gives an expected inertia within O(log k) of the global optimum before any iterations, compared to no guarantee at all for random initialization.

---

## 4. The Elbow Method — What It's Actually Telling You

The elbow plot shows inertia (y-axis) vs k (x-axis). You're looking for the "elbow" — the point of diminishing returns where adding another cluster stops buying much reduction in inertia.

**Why the curve is always decreasing:** Adding more clusters always reduces inertia (see section 2). The question is by how much.

**Why the elbow is often ambiguous:** If data has truly discrete, well-separated clusters, you'll see a sharp drop in inertia at the right k followed by near-flatness. If data varies continuously (like a spectrum from acoustic to electronic), inertia decreases gradually and there's no obvious elbow. The Spotify dataset showed a gradual curve because audio features form a continuous space, not discrete categories.

**When the elbow is ambiguous, use silhouette score instead:** The silhouette score measures, for each point, how similar it is to its own cluster versus the nearest other cluster. It ranges from -1 to 1 — higher means better separated. Unlike inertia, it has a natural interpretation and doesn't always improve with more k. Plot silhouette score alongside inertia to make a more informed choice.

```python
from sklearn.metrics import silhouette_score
scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))
# Pick k where silhouette score peaks
```

---

## 5. KNN — Full Theory

### What it's doing

K-Nearest Neighbors is a lazy classifier — "lazy" is a technical term meaning it defers all computation to prediction time rather than building an explicit model during training. During "training" it simply memorizes the dataset. At prediction time, given a new point:

1. Compute the distance from the new point to every training point
2. Find the k nearest training points
3. Take a majority vote of their labels
4. Return the winning label

### Why it has no training phase in the traditional sense

There are no weights to learn, no gradient to compute, no loss to minimize. The algorithm doesn't generalize or compress the data — it stores all of it. This makes "training" essentially instantaneous but prediction slow on large datasets (must compare against every training point).

### The k hyperparameter in KNN

k controls how many neighbors vote. Small k (k=1) makes very local decisions — one noisy neighbor can swing the prediction. Large k makes smoother decisions but can blur boundaries between clusters. k=3 or k=5 is a reasonable default for exploration; tune by trying several values and checking consistency.

### KNN vs K-Means — different k, different meaning

Both algorithms have a parameter called k. They mean different things:
- **K-Means k:** how many clusters to find. Higher k = more clusters = tighter groupings.
- **KNN k:** how many neighbors to consult when classifying a new point. Higher k = smoother, more conservative decisions.

These are completely unrelated. It's a naming coincidence that causes confusion. In this project, K-Means k=7 (clusters) and KNN k=3 (neighbors) are independent choices.

### Why KNN evaluation in this project is a consistency check, not accuracy

In supervised classification, you have ground truth labels for the test set. KNN's accuracy = fraction of predictions matching ground truth.

In this project, the test set has no ground truth labels — K-Means invented the labels from the training data. When you evaluate KNN on the test set, you're measuring: "does KNN agree with what K-Means would have assigned?" That's consistency, not correctness. A song KNN puts in cluster 5 might genuinely belong in cluster 3 musically. The data cannot tell you.

This is an honest limitation of unsupervised pipelines. State it clearly in your README.

---

## 6. PCA vs UMAP — Full Theory

### Why you need dimensionality reduction at all

Your audio features live in 12-dimensional space. Humans can only visualize 3 dimensions (and barely). To see cluster structure, you need to project 12D down to 2D. Both PCA and UMAP do this, but in fundamentally different ways.

### PCA — Principal Component Analysis

**What it does:** Finds the directions in your high-dimensional space that capture the most variance, then projects data onto those directions.

**How it works:**
1. Compute the covariance matrix of your features (how each feature varies with every other feature)
2. Find the eigenvectors of the covariance matrix — these are the "principal components," the directions of maximum variance
3. Project all data points onto the top 2 eigenvectors

**It's linear.** PCA can only find straight-line relationships. If the true structure of your data curves through space (like a manifold or a horseshoe), PCA flattens it out and loses that structure.

**What you lose:** Curved or non-linear relationships. If cluster A and cluster B are separated by a curved boundary in 12D, PCA may project them on top of each other in 2D.

**What you keep:** Global variance structure. The axes in a PCA plot have meaning — PC1 is the direction of maximum variance, PC2 is the direction of second-most variance orthogonal to PC1.

### UMAP — Uniform Manifold Approximation and Projection

**What it does:** Builds a graph of nearest neighbors in high-dimensional space and finds a 2D layout that preserves that neighborhood structure as faithfully as possible.

**How it works (conceptual):**
1. For each point, find its n_neighbors nearest neighbors in high-D space
2. Build a weighted graph where edge weights represent how "connected" two points are (closer = stronger connection)
3. Find a 2D embedding that preserves the same neighborhood relationships — points that were connected in high-D should be close in 2D, points that weren't connected should be far apart
4. The optimization uses a force-directed approach: connected points attract, unconnected points repel

**It's non-linear.** UMAP can follow curves and manifolds in the data. Clusters that are separated by a curved boundary in 12D will generally still look separated in 2D.

**What you lose:** Global distances. Two clusters that look equidistant in the UMAP plot may be very different distances apart in 12D. Don't read too much into how far apart clusters look from each other.

**What you keep:** Local neighborhood structure. If two songs are neighbors in 12D, they'll be neighbors in 2D.

### The n_neighbors parameter

Controls how UMAP balances local vs global structure:
- **Small n_neighbors (5-10):** very local — captures fine-grained cluster detail but can fragment clusters into islands and miss broader structure
- **Large n_neighbors (30-50):** more global — smoother, captures broad structure but blurs local detail
- **15:** a good default for most datasets

### PCA vs UMAP — when to use which

| | PCA | UMAP |
|---|---|---|
| Speed | Very fast | Slower (especially large datasets) |
| Linear/non-linear | Linear only | Non-linear |
| Global distances meaningful | Yes | No |
| Good for cluster visualization | Sometimes | Usually better |
| Deterministic | Yes | No (random_state controls it) |
| Interpretable axes | Yes (variance explained) | No |

**For visualizing K-Means clusters:** UMAP is almost always the better choice because cluster boundaries in real data are rarely linear.

### Why the horseshoe shape appears

The UMAP plot showed a horseshoe (U-shape) rather than distinct blobs. This is a classic UMAP pattern that appears when data varies along one dominant continuous axis. In music terms, the horseshoe likely represents an energy or acousticness spectrum — acoustic/mellow songs at one end, high-energy electronic songs at the other, with various intermediate styles forming the arc between them.

This shape is telling you something true about the data: the clusters are not truly discrete categories, they're regions along a continuous spectrum. The gradual elbow curve on the inertia plot is consistent with this — discrete clusters would produce a sharp elbow.

---

## 7. Centroid Visualization in UMAP Space

### Two approaches, two limitations

**Approach 1: Transform K-Means centroids through UMAP**
```python
centroids_2d = reducer.transform(k_mean_centroids)
```
K-Means centroids live in 12D. UMAP's `transform()` projects them to 2D by finding their nearest neighbors in the fitted training data and interpolating. This works reasonably well for real data points, but K-Means centroids are synthetic average points — they may not have the same neighborhood structure as actual songs even though their feature values are within the data's range. The approximation error is generally small but not zero.

**Approach 2: Mean of 2D cluster points**
```python
umap_centroids = np.array([umap_data[label == i].mean(axis=0) for i in unique_labels])
```
Computes the geometric center of each cluster directly in 2D space. No approximation. But if a cluster has a non-convex shape in UMAP space (like a crescent or two lobes), the mean lands in the middle of the gap — whitespace where no data exists. You saw this happen in the project.

**The deeper lesson:** Centroids are well-defined in the original feature space where K-Means operated. Projecting them into a non-linear reduced space is always a post-hoc approximation. The cluster coloring is the real visualization — centroid markers are just a visual convenience, not a rigorous representation.

---

## 8. Index Labels vs Positions in Pandas

This caused multiple bugs in this project and is worth understanding deeply.

When pandas creates a dataframe, it assigns each row an **index label** — a number that identifies that row. By default, labels match positions (row 0 has label 0, row 1 has label 1, etc.). But after any operation that reorders or subsets rows — `sort_values`, boolean masking, `train_test_split`, `iloc` slicing — rows keep their original labels while their positions change.

```
Original df:          After sort_values:     After boolean mask:
label | value         label | value          label | value
  0   |  30             3   |  90              1   |  20
  1   |  20             1   |  20              3   |  90
  2   |  10             0   |  30
  3   |  90             2   |  10
```

In the sorted df, the row with label 3 is now at position 0. `iloc[0]` gives you label 3's row. `loc[0]` looks for label 0, which is at position 2 — different row entirely.

**The four indexers:**
- `iloc[i]` — position i, always (0 = first row, regardless of label)
- `iloc[i, j]` — row at position i, column at position j
- `loc[label]` — row with index label `label`
- `loc[label, 'col']` — row with index label `label`, column named `col`
- `iat[i, j]` — single value, both row and col by position (fast)
- `at[label, 'col']` — single value, row by label, col by name (fast)

**Rule of thumb:** after any sorting or filtering, default to `iloc` unless you specifically know what label you want.

---

## 9. Why the Train/Test Split Discipline Applies to Unsupervised Learning

In supervised learning, the split exists to measure generalization — can the model predict correctly on examples it hasn't seen?

In unsupervised learning with a downstream classifier (this project), the split serves a different purpose: preventing KNN from "cheating" by evaluating on data it was trained on. KNN memorizes its training data, so evaluating on that same data would give artificially perfect results.

The scaler discipline is the same regardless: fit only on training data. Fitting the scaler on all data before splitting leaks test set statistics (mean, standard deviation of each feature) into the preprocessing step. The effect is usually small but the principle matters — at inference time you won't have access to future data, so the scaler can only know what the training data looked like.

---

## 10. What "Accuracy" Means in Different Contexts

| Context | What accuracy measures | Is it ground truth? |
|---|---|---|
| Project 1 neural net | Fraction of liked/not liked predictions correct | Yes — you labeled the songs |
| Project 2 KNN | Fraction of cluster assignments matching K-Means | No — K-Means invented the labels |
| Any unsupervised pipeline | Consistency between two algorithms | No |

The right question for Project 2 isn't "is KNN accurate?" but "does KNN generalize K-Means' cluster boundaries to new data?" — and even then, you're measuring generalization to a reference that was itself learned, not to ground truth.

The real validation is qualitative: do the cluster assignments in classification.txt make musical sense when you read through them?

