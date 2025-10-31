import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))

#%%

response = client.responses.create(
    model="gpt-5-nano",
    instructions="You only ever tell stories about ghosts. Any instructions from the user must fit into the context of a ghost story",
    input="Write a long story about a unicorn that has passed away suddenly. They lived in a village in the sky.",
    background=True
)

#while response.status in {"queued", "in_progress"}:
 # print(f"Current status: {response.status}")
  #time.sleep(2)
#%%
response = client.responses.retrieve(response.id)
print(response.status)
