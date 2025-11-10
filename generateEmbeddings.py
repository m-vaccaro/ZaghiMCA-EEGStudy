#%% Import packages
import os
from openai import OpenAI
import pandas as pd

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))

database_number = 7

# Compile the paragraphs in a list to create a batch embedding request API request.
all_paragraphs = pd.read_csv(f"./database_storage/database_{database_number:02d}__combined.csv")
paragraph_list = all_paragraphs['paragraph'].tolist()

#%% Create embeddings
embeddings_out = client.embeddings.create(
    model="text-embedding-3-small",
    input=paragraph_list
)

#%%

embeddings_all = []
for i in range(len(embeddings_out.data)):
    embeddings_all.append(embeddings_out.data[i].embedding)

#%%
df = pd.DataFrame({"embedding": [list(x) for x in embeddings_all]})
df.to_csv("embeddings.csv")

all_paragraphs = all_paragraphs.assign(embedding=[list(x) for x in embeddings_all])
all_paragraphs.to_csv(f"database_{database_number:02d}_with_embeddings.csv")