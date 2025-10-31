import os
import json
from openai import OpenAI
import time
import pandas as pd
from io import StringIO

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))

#%% Define messages
sysMsg = "You are generating a large library of single-paragraph texts to maximize semantic, stylistic, and lexical diversity for EEG reading experiments. Hard constraints:\n-Exactly one paragraph per item, 100 to 150 words, no lists or headings.\n-No personal data, hate, sexual content, medical or legal advice, or partisan politics; keep topics neutral and classroom-safe.\n-Each item must be meaningfully different from earlier items in topic, rhetorical mode, tone, register, pacing, and lexical choice; avoid repeated phrasings and cliches.\n-Do not self-reference, do not explain what you’re doing.\n-Output as JSON Lines, one object per paragraph, matching the provided JSON Schema.\n\nDiversity axes to systematically cover:\n-Domain (rotate evenly): {life_sciences, physical_sciences, engineering, computing, humanities, social_sciences, everyday_scenarios, nature_travel, arts_culture}\n-Rhetorical mode: {narrative, expository, descriptive, process_explanation, persuasive}\n-Tone / register: {plain, formal, technical, playful, reflective, conversational}\n-Reading level (targeted): {Grade8, Grade12, Undergraduate, Graduate}. Approximate by sentence length, vocabulary, and syntax\n-Style knobs (choose a setting each time): sentence_length={short, mixed, long}, figurative_language={none, low, medium, high}, concreteness={abstract, mixed, concrete}, viewpoint={1st, 2nd, 3rd}, temporal_focus={past, present, future}.\n\nQuality rules:\n-Keep facts generic or obviously illustrative; avoid controversial specifics.\n-Prefer fresh imagery and varied verbs; rotate discourse markers and clause structure.\n-Ensure coherence in ~120 words; no dangling references or unexplained jargon."

usrMsg = "Generate exactly 5 items. Cycle through the domain × mode × tone × reading_level grid before repeating any combination. Randomize style knobs each time. For each item, invent a new micro-scenario and vocabulary set; avoid reusing salient bigrams (other than stop-words). Keep each paragraph between 100 and 150 words long. Output one JSON object per line as specified, with no extra text."

with open("paragraph_output_schema.json", "r") as jsonFile:
    output_schema = json.load(jsonFile)

# print(output_schema)

#%%
response = client.responses.create(
    model="gpt-4o",
    instructions=sysMsg,
    input=usrMsg,
    background=True,
    text={
        "format": {"type": "json_schema", "name": "math_response", "schema": output_schema, "strict": True}
    }
)

timeStart = time.time()
while response.status in {"queued", "in_progress"}:
    print(f"Current status: {response.status}")
    time.sleep(5)
    response = client.responses.retrieve(response.id)
    print(response.status)
    print(f"Elapsed time: {time.time() - timeStart}\n\n---\n")

#%%
database_name = "database_04"

data = json.loads(response.output_text)  # parse to a dict (validates JSON)
with open(f"{database_name}.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Save to a CSV
data = json.loads(response.output_text)

df = pd.json_normalize(data['items'])   # Normalize the items array (this flattens the nested 'style' dict)
df.to_csv(f"{database_name}.csv", index=False)    # save file