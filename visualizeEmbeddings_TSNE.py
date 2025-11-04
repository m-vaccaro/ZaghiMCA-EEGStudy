# Goal: visualize embeddings with t-SNE.

# Recommendations for t-SNE from https://distill.pub/2016/misread-tsne/:
#   Perplexity should be less than the total number of points
#   Iterate until long-term stability

# Recommendations for t-SNE from sklearn website:
#   - Use another dimensionality reduction method (e.g., PCA for dense data) to reduce the number of dimensions to a
#   reasonable amount (e.g., 50) if the number of features is very high. This suppresses noise and speeds up computation
#   of pairwise distances between samples.
#   - Perplexity must be less than the total number of points
#   - The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
#   Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.
#   - If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from
#   its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few
#   outliers. If the cost function gets stuck in a bad local minimum increasing the learning rate may help.


#%% Decide on a schema to color data points by & Import packages and data
# For starters, try out coloring by subject area/domain:
#   life_sciences, physical_sciences, engineering, computing, humanities, social_sciences, everyday_scenarios,
#   nature_travel, arts_culture

# Specify colors based on order listed above
colors_domain = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:cyan",
                 "tab:olive", "tab:brown"]

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from ast import literal_eval

database = pd.read_csv("./database_storage/database_with_embeddings__50Texts.csv")
embedding_matrix = np.array(database.embedding.apply(literal_eval).to_list())

domains = ["life_sciences","physical_sciences","engineering","computing",
    "humanities","social_sciences","everyday_scenarios","nature_travel","arts_culture"]

dtype = pd.api.types.CategoricalDtype(categories=domains, ordered=True)
database["domain_id"] = database["domain"].astype(str).str.strip().str.lower().astype(dtype).cat.codes  # unknown => -1


#%% Set hyperparameters, create a t-SNE model, and transform the data

# Specify hyperparameters:
final_dims = 2 # Number of dimensions for the final transformation
perplex = 15 # Perplexity
r_state = 17 # Set a random state for repeatable results. NOTE: Different initializations can result in different local minima, so test stability
initialization = 'random' # Initialization of embedding (random or pca). PCA initialization is usually more globally stable.
l_rate = 10 # Learning rate

# Create a t-SNE model with scikit learn
tsne = TSNE(n_components=final_dims, perplexity=perplex, init=initialization, learning_rate=l_rate, random_state=r_state)

# Fit the t-SNE model
embeddings_tsne = tsne.fit_transform(embedding_matrix)
print(f'Fit successful with shape {embeddings_tsne.shape}.')


#%% Create plots of t-SNE transformation
paragraph_domains = pd.read_csv("./database_storage/database_with_embeddings__50Texts.csv").domain.to_list()

x_data = [x for x,y in embeddings_tsne]
y_data = [y for x,y in embeddings_tsne]
color_indices = database.domain_id

colormap = matplotlib.colors.ListedColormap(colors_domain)

plt.scatter(x_data, y_data, c=color_indices, cmap=colormap, alpha=0.8)
plt.title(f"{final_dims}-D t-SNE Transformation with perplexity = {perplex} \n and {initialization} initial state (l_rate = {l_rate})")

plt.show()