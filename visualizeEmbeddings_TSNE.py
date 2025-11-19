# Goal: visualize embeddings with t-SNE and color points by a chosen category.

#%% Choose which category to color points by
# Logical options:
#   "domain", "mode", "tone", "reading_level",
#   "sentence_length", "figurative_language", "concreteness", "viewpoint", "temporal_focus"
color_by = "temporal_focus"  # <-- change this to recolor

#%% Imports and data loading
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from ast import literal_eval

database_number = 11
database_name = "gpt4o-full"

database = pd.read_csv(
    f"./database_storage/database_{database_number:02d}-{database_name}__embeddings.csv"
)

# Embeddings column assumed to be a stringified list (e.g. "[0.1, 0.2, ...]")
embedding_matrix = np.array(database.embedding.apply(literal_eval).to_list())

# Optional: load paragraph domains (not used in plotting but kept from original code)
paragraph_domains = pd.read_csv(
    "./database_storage/database_with_embeddings__50Texts.csv"
).domain.to_list()

#%% Map logical category names to actual DataFrame column names

column_map = {
    "domain": "domain",
    "mode": "mode",
    "tone": "tone",
    "reading_level": "reading_level",
    # style.* fields in your dataframe:
    "sentence_length": "style.sentence_length",
    "figurative_language": "style.figurative_language",
    "concreteness": "style.concreteness",
    "viewpoint": "style.viewpoint",
    "temporal_focus": "style.temporal_focus",
}

# This is the actual column we will use:
color_column = column_map[color_by]

#%% Define category enums and color palettes

category_values = {
    "domain": [
        "life_sciences", "physical_sciences", "engineering", "computing",
        "humanities", "social_sciences", "everyday_scenarios", "nature_travel", "arts_culture"
    ],
    "mode": [
        "narrative", "expository", "descriptive", "process_explanation", "persuasive"
    ],
    "tone": [
        "plain", "formal", "technical", "playful", "reflective", "conversational"
    ],
    "reading_level": [
        "Grade8", "Grade12", "Undergraduate", "Graduate"
    ],
    "sentence_length": ["short", "mixed", "long"],
    "figurative_language": ["none", "low", "medium", "high"],
    "concreteness": ["abstract", "mixed", "concrete"],
    "viewpoint": ["1st", "2nd", "3rd"],
    "temporal_focus": ["past", "present", "future"],
}

category_colors = {
    "domain": [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:pink", "tab:cyan", "tab:olive", "tab:brown"
    ],
    "mode": ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
    "tone": ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"],
    "reading_level": ["tab:blue", "tab:orange", "tab:green", "tab:red"],
    "sentence_length": ["tab:blue", "tab:orange", "tab:green"],
    "figurative_language": ["tab:blue", "tab:orange", "tab:green", "tab:red"],
    "concreteness": ["tab:blue", "tab:orange", "tab:green"],
    "viewpoint": ["tab:blue", "tab:orange", "tab:green"],
    "temporal_focus": ["tab:blue", "tab:orange", "tab:green"],
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
perplex = 16    # Perplexity
r_state = 17    # Random state for repeatable results
initialization = "random"  # 'random' or 'pca'
l_rate = 40     # Learning rate

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
