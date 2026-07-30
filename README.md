# Music Taste Mapper

**Project 2 of 12** in a structured ML/AI skill-building roadmap. Every line of core logic is written by hand — no AI-generated implementations.

---

## What This Does

Takes a dataset of Spotify songs and discovers the hidden structure of music taste using unsupervised learning. Rather than predicting whether a song would be liked (that's Project 1), this project asks a different question: **what natural groupings exist in the audio feature space, and what do those groupings represent musically?**

The result is a system that can take any song's audio features and tell you which "taste cluster" it belongs to — not based on genre labels, but purely based on what the music sounds like.

---

## The Core Question Each Model Answers

This project is most useful when understood alongside Project 1. Three models, three different questions:

- **Neural net (Project 1):** "Would I like this song?" — supervised, uses preference labels, learns a decision boundary between liked and not liked.
- **K-means (this project):** "What natural groupings exist in this music?" — fully unsupervised, no labels, finds compact regions in audio feature space.
- **KNN (this project):** "Given a new song, which cluster does it most resemble?" — lazy classifier, no training phase, memorizes the training data and answers by finding nearest neighbors.

K-means discovers the structure. KNN exploits that structure for fast lookup on new songs. The neural net ignores structure entirely in favor of preference prediction. These are genuinely different questions with genuinely different tools.

---

## Dataset

TidyTuesday Spotify Songs dataset (January 2020). 32,833 songs pulled from Spotify playlists across six genres.

**Audio features used:**
- `danceability`, `energy`, `key`, `loudness`, `mode`
- `speechiness`, `acousticness`, `instrumentalness`, `liveness`
- `valence`, `tempo`, `duration_ms`

Source: [rfordatascience/tidytuesday](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv)

---

## Approach

### 1. Data Splitting
80/20 train/test split before any preprocessing. K-means and the StandardScaler are fit exclusively on the 80% training set. The 20% test set is held out to evaluate KNN.

### 2. Preprocessing
StandardScaler normalization fit only on training data. Critical because K-means uses Euclidean distance — features with larger scales (e.g. loudness in dB vs. danceability 0–1) would otherwise dominate the distance calculation regardless of actual musical significance.

### 3. Elbow Method
K-means run for k=2 through k=10. Inertia (sum of squared distances from each point to its assigned centroid) plotted against k. The curve is gradual with no sharp elbow — common in music data where audio features form a continuous spectrum rather than perfectly discrete groups. k=7 selected as the point where marginal inertia reduction begins to flatten.

### 4. K-Means Clustering
Final K-means fit with k=7 using k-means++ initialization (smarter centroid seeding that reduces the chance of getting stuck in local optima) and n_init=10 (runs 10 times with different seeds, keeps the best result).

### 5. UMAP Visualization
12-dimensional scaled features projected to 2D using UMAP for visualization. UMAP preserves local neighborhood structure — points that are close in 12D stay close in 2D — making it better than PCA for visualizing cluster separation. The resulting plot shows a horseshoe shape, suggesting the dataset has one dominant continuous axis of variation (likely energy) rather than perfectly discrete groups.

**Note on centroids in UMAP space:** Two approaches were explored — transforming the 12D K-means centroids through UMAP's transform(), and computing centroids directly as the mean of each cluster's 2D UMAP coordinates. The second approach is more principled but can place centroids in whitespace when a cluster has a non-convex shape in 2D. This is a known limitation of visualizing high-dimensional centroids in reduced space.

### 6. Manual Cluster Labeling
Representative songs identified by sorting each cluster by track popularity and inspecting audio features. Cluster labels assigned based on the feature profile of top songs:

| Cluster | Label | Representative Song | Key Features |
|---------|-------|-------------------|--------------|
| 0 | Mid-energy pop/R&B | Circles — Post Malone | Moderate energy (0.76), high danceability |
| 1 | Acoustic/mellow pop | Memories — Maroon 5 | High acousticness (0.837), low energy (0.32) |
| 2 | Speechy mid-energy | ROXANNE — Arizona Zervas | High speechiness (0.148), high liveness (0.46) |
| 3 | High energy EDM/dance | Baila Conmigo — Dayvi | Near-max energy (0.972), high instrumentalness (0.465) |
| 4 | Latin/reggaeton | Tusa — KAROL G | High danceability (0.803), high speechiness (0.298) |
| 5 | High energy mainstream pop | Blinding Lights — The Weeknd | High energy (0.796), fast tempo (171 BPM) |
| 6 | Mixed acoustic/electronic pop | Dance Monkey — Tones and I | High acousticness (0.692), high danceability (0.824) |

### 7. KNN Classification
KNN trained on (X_k_means_scaled → K-means cluster labels). k=3 neighbors. Evaluated on the held-out 20% test set.

**Important caveat on evaluation:** There are no ground truth labels for the test set — this is unsupervised learning. The "accuracy" metric here measures KNN's consistency with K-means, not whether the cluster assignments are musically correct. A song KNN places in cluster 5 might genuinely belong in cluster 3 musically — the data cannot tell you. The real validation is qualitative: do the classifications in classification.txt make musical sense?

---

## How to Run

```bash
# Clone the repo
git clone <repo-url>
cd SongClusters

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

The elbow plot will display first. Enter your chosen k value when prompted. The UMAP visualization and classification output will follow. Results are written to `classification.txt`.

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
umap-learn
joblib
```

---

## Key Learnings

### Why unsupervised learning is different from supervised
No labels means no loss function to minimize against ground truth. K-means minimizes inertia — a geometric objective — not a prediction error. The algorithm invents its own "labels" (cluster assignments) as output, which then become the supervision signal for KNN.

### Why scaling matters for distance-based algorithms
K-means and KNN both rely on Euclidean distance. Features with larger numeric ranges dominate distance calculations regardless of musical significance. StandardScaler brings all features to the same scale before any distance is computed.

### The scaler must be fit only on training data
Fitting the scaler on the full dataset before splitting leaks test set statistics (mean, standard deviation) into preprocessing. The scaler is fit on the 80% training set only, then applied identically to both splits.

### UMAP vs PCA
PCA is linear — it finds the directions of maximum variance and projects onto them. UMAP is non-linear — it builds a nearest-neighbor graph in high-dimensional space and finds a 2D layout that preserves that neighborhood structure. For visualizing clusters, UMAP is generally preferred because cluster structure is often non-linear.

### The horseshoe shape
The UMAP plot shows a horseshoe rather than distinct blobs. This is a signal that the data varies along one dominant continuous axis (likely energy) rather than falling into perfectly discrete groups. The gradual elbow curve confirms this — if there were truly distinct clusters, inertia would drop sharply at the right k.

### KNN has no training phase in the traditional sense
KNN memorizes the training data. At prediction time it computes distances from the new point to all training points and takes a majority vote among the k nearest neighbors. There are no weights to learn, no gradient to compute. This makes it fast to "train" but slow to predict on large datasets.

---

## Limitations

- **No ground truth labels** — cluster quality is validated qualitatively, not quantitatively. Silhouette score would be a more rigorous diagnostic (flagged for future addition).
- **Duplicate songs** — the dataset contains songs appearing in multiple playlists, so the same song can appear in a cluster multiple times. Deduplication by track_id before clustering would produce cleaner results.
- **k=7 is a judgment call** — the elbow curve is ambiguous. Different values of k produce meaningfully different cluster structures and labels.
- **UMAP distorts global distances** — clusters that look equidistant in the 2D plot may be very different distances apart in 12D space. The visualization is for intuition, not precise distance reading.

---

## Project Roadmap

This is Project 2 of 12. Each project is an independent repo targeting a specific set of ML/AI skills.

<!-- ADD LINK TO FULL ROADMAP -->
