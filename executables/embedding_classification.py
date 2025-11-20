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

# STEP 1. Load embeddings database

database_number = 13
database_name = "gpt5_1-full"
embedding_size = "large"

df = pd.read_csv(f"../database_storage/database_{database_number:02d}-{database_name}__embeddings-{embedding_size}.csv")

# Convert embeddings to numeric matrix
X = np.array(df["embedding"].apply(literal_eval).to_list())
print("Embeddings shape:", X.shape)

#%% STEP 2. Define which labels to test

label_columns = [
    "topic_hint",
    "genre",
    "difficulty",
    "coherence_predictability",
    "emotional_valence",
    "concreteness",
    "tone",
]

#%% STEP 3. Set up classifier and cross-validation

# Simple pipeline: (optional PCA) + multinomial logistic regression
use_pca = True
n_pca_components = 50  # adjust as needed; must be < n_samples

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
