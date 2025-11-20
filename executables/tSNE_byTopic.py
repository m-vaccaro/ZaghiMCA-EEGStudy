# Goal: visualize embeddings with t-SNE and color points by a chosen category,
#       one topic_hint at a time.

#%% Choose which category to color points by
# Logical options:
#   "genre", "difficulty", "coherence_predictability",
#   "emotional_valence", "concreteness", "tone"
color_by = "tone"  # <-- change this to recolor

#%% Imports and data loading
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from ast import literal_eval

database_number = 13
database_name = "gpt5_1-full"
embedding_size = "large"

database = pd.read_csv(
    f"../database_storage/database_{database_number:02d}-{database_name}__embeddings-{embedding_size}.csv"
)

# Embeddings column assumed to be a stringified list (e.g. "[0.1, 0.2, ...]")
embedding_matrix = np.array(database.embedding.apply(literal_eval).to_list())

#%% Map logical category names to actual DataFrame column names

column_map = {
    "genre": "genre",
    "difficulty": "difficulty",
    "coherence_predictability": "coherence_predictability",
    "emotional_valence": "emotional_valence",
    "concreteness": "concreteness",
    "tone": "tone",
    "topic_hint": "topic_hint",
}

color_column = column_map[color_by]

#%% Define category enums and color palettes

category_values = {
    "topic_hint": [
        "life_sciences", "physical_sciences", "engineering", "computing"
    ],
    "genre": [
        "narrative", "expository"
    ],
    "difficulty": [
        "high", "medium", "low"
    ],
    "coherence_predictability": [
        "high_coherence_high_predictability",
        "low_coherence",
        "high_coherence_low_predictability"
    ],
    "emotional_valence": ["negative", "positive", "neutral"],
    "concreteness": ["abstract", "mixed", "concrete"],
    "tone": ["reflective", "technical", "plain"]
}

category_colors = {
    "topic_hint": ["tab:blue", "tab:orange", "tab:green", "tab:red"],
    "genre": ["tab:blue", "tab:orange"],
    "difficulty": ["tab:blue", "tab:orange", "tab:green"],
    "coherence_predictability": ["tab:blue", "tab:orange", "tab:green"],
    "emotional_valence": ["tab:blue", "tab:orange", "tab:green"],
    "concreteness": ["tab:blue", "tab:orange", "tab:green"],
    "tone": ["tab:blue", "tab:orange", "tab:green"]
}

labels = category_values[color_by]
colors_for_category = category_colors[color_by]

#%% Get list of topics to loop over
topics = ["life_sciences", "physical_sciences", "engineering", "computing"]
# or: topics = sorted(database["topic_hint"].unique())

#%% Loop over topics and make a t-SNE plot for each

for topic in topics:
    # Subset dataframe and embeddings to this topic
    mask = database["topic_hint"] == topic
    df_topic = database.loc[mask].reset_index(drop=True)
    emb_topic = embedding_matrix[mask]

    n_samples = emb_topic.shape[0]
    if n_samples < 10:
        print(f"Skipping topic '{topic}' (too few samples: {n_samples})")
        continue

    # Encode the chosen category within this topic
    dtype = pd.api.types.CategoricalDtype(categories=labels, ordered=True)
    df_topic[f"{color_by}_id"] = (
        df_topic[color_column]
        .astype(str)
        .str.strip()
        .astype(dtype)
        .cat.codes
    )

    color_indices = df_topic[f"{color_by}_id"]

    # ----- PCA -> t-SNE for this topic -----
    # PCA to denoise / compress first (up to 50 PCs, but no more than n_samples-1)
    pca_components = min(50, n_samples - 1)
    pca = PCA(n_components=pca_components, random_state=0)
    emb_pca = pca.fit_transform(emb_topic)

    # t-SNE hyperparams
    final_dims = 2
    # perplexity must be < n_samples; use min(30, n_samples//3) as a safe heuristic
    perplex = min(30, max(5, n_samples // 3))
    tsne = TSNE(
        n_components=final_dims,
        perplexity=perplex,
        init="pca",
        learning_rate="auto",
        random_state=17,
    )

    emb_tsne = tsne.fit_transform(emb_pca)
    print(f"[{topic}] t-SNE shape: {emb_tsne.shape}, n={n_samples}, perplexity={perplex}")

    # ----- Plot -----
    x_data = emb_tsne[:, 0]
    y_data = emb_tsne[:, 1]

    colormap = matplotlib.colors.ListedColormap(colors_for_category)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, c=color_indices, cmap=colormap, alpha=0.85)

    plt.title(
        f"{final_dims}-D t-SNE (topic={topic}, perplexity={perplex})\n"
        f"Database: {database_name} | Colored by: {color_by}",
        fontsize=10,
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Legend
    legend_elements = []
    for idx, label in enumerate(labels):
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=colors_for_category[idx],
                markeredgecolor="none",
                label=label,
            )
        )

    if (color_indices == -1).any():
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor="lightgray",
                markeredgecolor="none",
                label="unknown",
            )
        )

    plt.legend(handles=legend_elements, title=color_by, loc="best", fontsize=8, title_fontsize=9)
    plt.tight_layout()
    plt.show()
