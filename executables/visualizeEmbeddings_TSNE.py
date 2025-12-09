# Goal: visualize embeddings with t-SNE and color points by a chosen category.

#%% Choose which category to color points by
# Logical options:
#   "genre", "difficulty", "coherence_predictability",
#   "emotional_valence", "concreteness", "tone", "topic_hint"
list_categories = ["genre", "difficulty", "coherence_predictability", "emotional_valence", "concreteness", "tone", "topic_hint"]  # <-- change this to recolor

#%% Imports and data loading
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from ast import literal_eval

database_number = 17
database_name = "gpt5_1-full-25_to_50_words"
embedding_size = "large"

database = pd.read_csv(
    f"../database_storage/database_{database_number:02d}-{database_name}__embeddings-{embedding_size}.csv"
)

# Embeddings column assumed to be a stringified list (e.g. "[0.1, 0.2, ...]")
embedding_matrix = np.array(database.embedding.apply(literal_eval).to_list())

# Optional: load paragraph domains (not used in plotting but kept from original code)
paragraph_domains = pd.read_csv(
    "../database_storage/old_databases/database_with_embeddings__50Texts.csv"
).domain.to_list()

#%% Map logical category names to actual DataFrame column names

column_map = {
    "genre": "genre",
    "difficulty": "difficulty",
    "coherence_predictability": "coherence_predictability",
    # style.* fields in your dataframe:
    "emotional_valence": "emotional_valence",
    "concreteness": "concreteness",
    "tone": "tone",
    "topic_hint": "topic_hint",
}

for color_by in list_categories:
    # This is the actual column we will use:
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
            "high_coherence_high_predictability", "low_coherence", "high_coherence_low_predictability"
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

    # Convert the chosen category column to categorical codes
    dtype = pd.api.types.CategoricalDtype(categories=labels, ordered=True)
    database[f"{color_by}_id"] = (
        database[color_column]      # <--- note: using the mapped column name
        .astype(str)
        .str.strip()
        .astype(dtype)
        .cat.codes                  # unknown/missing -> -1
    )

    #%% Set hyperparameters, create a t-SNE model, and transform the data

    final_dims = 2  # Number of dimensions for the final transformation
    perplex = 30    # Perplexity
    r_state = 17    # Random state for repeatable results
    initialization = "random"  # 'random' or 'pca'
    l_rate = "auto"     # Learning rate

    tsne = TSNE(
        n_components=final_dims,
        perplexity=perplex,
        init=initialization,
        learning_rate=l_rate,
        random_state=r_state,
    )

    embeddings_tsne = tsne.fit_transform(embedding_matrix)
    print(f"Fit successful with shape {embeddings_tsne.shape}.")

    #%% Create plots of t-SNE transformation

    x_data = [x for x, y in embeddings_tsne]
    y_data = [y for x, y in embeddings_tsne]
    color_indices = database[f"{color_by}_id"]

    colormap = matplotlib.colors.ListedColormap(colors_for_category)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, c=color_indices, cmap=colormap, alpha=0.8)

    plt.title(
        f"{final_dims}-D t-SNE (perplexity={perplex}, init={initialization}, l_rate={l_rate})\n"
        f"Database: {database_name} | Colored by: {color_by}",
        fontsize=10,
    )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Build legend from labels & colors
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

    # Optional: add "unknown" if any -1 codes exist
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
