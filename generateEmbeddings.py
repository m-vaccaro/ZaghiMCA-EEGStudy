#%% Import packages
import os
import openai
from openai import OpenAI
import pandas as pd

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))

# Compile the paragraphs in a list to create a batch embedding request API request.
all_paragraphs = pd.read_csv("./database_storage/database_06__50Texts.csv")
paragraph_list = all_paragraphs['paragraph'].tolist()

#%% Create embeddings
embeddings_out = client.embeddings.create(
    model="text-embedding-3-large",
    input=paragraph_list
)

#%%

embeddings_all = []
for i in range(len(embeddings_out.data)):
    embeddings_all.append(embeddings_out.data[i].embedding)

#%%
df = pd.DataFrame({"embeddings": embeddings_all})
df.to_csv("embeddings.csv", index=False)