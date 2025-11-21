#%% State purpose and import required packages
# Goal: We previously generated texts using GPT 5.1 across 4 domains and using several different style knobs:
#   topic_hint: life_sciences, physical_sciences, engineering, and computing
#   genre: narrative, expository
#   difficulty: low, medium, high
#   coherence_predictability: high_coherence_high_predictability, high_coherence_low_predictability, low_coherence
#   emotional_valence: negative, neutral, positive
#   concreteness: abstract, mixed, concrete
#   tone: plain, technical, reflective
#
# The main task now is to ask: "Given only the text embeddings, how well can a simple classifier guess the above labels?"
#   For each category (e.g., genre), is the label (e.g., narrative) strongly encoded in the embedding
#   (accuracy >> chance), moderately encoded (just above chance), or not encoded (~chance)?
#
# **Later on we will be going from text_embedding to the EEG feature vector. If we can't decode a variable of interest
# from the embedding itself, we might not expect it to be found in the EEG data (or at least the regression model from
# text to EEG is not going to work well for that variable since it isn't even encoded in the embedding!).
#
# **Basically, we are looking to answer the question: "What information (that we care about) is **actually** encoded in
# our text embeddings?"

#%% Implementation
import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# STEP 1. Load embeddings database and set other parameters

database_number = 13
database_name = "gpt5_1-full"
embedding_size = "large"

# Parameters for logistic regression
use_pca = True
min_cum_variance_pca = 0.8

df = pd.read_csv(f"../database_storage/database_{database_number:02d}-{database_name}__embeddings-{embedding_size}.csv")

# Convert embeddings to numeric matrix
X = np.array(df["embedding"].apply(literal_eval).to_list())
print("Embeddings shape:", X.shape)

# Define which labels to test

label_columns = [
    "topic_hint",
    "genre",
    "difficulty",
    "coherence_predictability",
    "emotional_valence",
    "concreteness",
    "tone",
]

#%% STEP 2. Inspect PCA explained variance (global, across all texts)

# Choose an upper limit on number of components to inspect
max_components = min(X.shape[1], X.shape[0] - 1)
print(f"Fitting PCA with up to {max_components} components for variance plot...")

pca_plot = PCA(n_components=max_components, random_state=0)
pca_plot.fit(X)

explained_var_ratio = pca_plot.explained_variance_ratio_
cum_explained = np.cumsum(explained_var_ratio)

# Print a quick textual summary
print("\nFirst 10 components' explained variance ratio:")
for i, ev in enumerate(explained_var_ratio[:10], start=1):
    print(f"PC{i:2d}: {ev:.4f}")

print("\nCumulative explained variance at some key points:")
for k in [10, 20, 50, 100]:
    if k <= max_components:
        print(f"  {k:3d} PCs: {cum_explained[k-1]:.4f}")

# Decide how many PCA components to use in the logistic regression based on the cumulative explained variance:
n_pca_components = np.searchsorted(cum_explained, min_cum_variance_pca) + 1  # +1 because indices start at 0

print(f"\nNumber of PCs to reach >= {min_cum_variance_pca:.0%} variance: {n_pca_components}")
print(f"Cumulative variance at {n_pca_components} PCs: {cum_explained[n_pca_components-1]:.4f}")

# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(1, max_components + 1),
    cum_explained,
    marker="o",
    markersize=2,
    linewidth=1,
)
plt.axvline(50)
plt.axhline(0.8)
plt.xlabel("Number of PCA components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative explained variance of text embeddings")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% STEP 3. Set up classifier and cross-validation

# Note: While "multinomial" is not explicitly specified in the logistic regression, sklearn automatically does
# multinomial for n_classes >= 3.
if use_pca:
    clf = Pipeline([
        ("pca", PCA(n_components=min(n_pca_components, X.shape[0] - 1), random_state=0)),
        ("logreg", LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=2000,
            n_jobs=-1
        )),
    ])
else:
    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        n_jobs=-1
    )

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Scorers
acc_scorer = "accuracy"
bal_acc_scorer = make_scorer(balanced_accuracy_score)

#%% STEP 4. Loop over labels

for col in label_columns:
    y_raw = df[col].astype(str).str.strip()

    # Drop rows with missing/unknown labels if needed
    mask = (y_raw != "nan") & (y_raw != "")
    X_sub = X[mask]
    y_sub_raw = y_raw[mask]

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_sub_raw)

    n_classes = len(le.classes_)
    if n_classes < 2:
        print(f"\n[{col}] Skipping: only {n_classes} class present.")
        continue

    print(f"\n=== Label: {col} ===")
    print("Classes:", list(le.classes_))
    print("Class counts:", y_sub_raw.value_counts().to_dict())

    # Compute cross-validated accuracy
    acc_scores = cross_val_score(clf, X_sub, y, cv=cv, scoring=acc_scorer)
    bal_acc_scores = cross_val_score(clf, X_sub, y, cv=cv, scoring=bal_acc_scorer)

    chance_level = 1.0 / n_classes

    print(f"Chance level: {chance_level:.3f}")
    print(f"Accuracy: mean={acc_scores.mean():.3f}, std={acc_scores.std():.3f}")
    print(f"Balanced accuracy: mean={bal_acc_scores.mean():.3f}, std={bal_acc_scores.std():.3f}")
