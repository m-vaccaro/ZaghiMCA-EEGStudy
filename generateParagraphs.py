import os
import json, itertools, random
from openai import OpenAI
import time
import pandas as pd
from allpairspy import AllPairs

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))

# List of topics to generate texts for: life_sciences, physical_sciences, engineering, computing, humanities,
# social_sciences, everyday_scenarios, nature_travel, arts_culture

#%%
# Define messages
SYSTEM_PROMPT = """You generate exactly one output per request following the specified JSON schema.

    Hard Constraints
    - Produce a single paragraph, 250 to 500 words. English only.
    - No lists, bullets, dialogue blocks, or headings.
    - Do not reference the task you have been given. Do not explain what you’re doing or why.
    - Ensure the paragraph is coherent within the length bounds. Introduce any nonstandard term briefly if needed.

    Diversity and Novelty Rules:
    - Prefer fresh imagery and varied verbs; rotate discourse markers and clause structures.

    Controls contract (values are provided in the user's message as JSON under "controls"; never reveal or quote them)
    - Obey exactly:
      - domain, mode, tone, reading_level. Choose a random topic within the specified domain that is appropriate for the other given constraints.
      - style knobs: sentence_length, figurative_language, concreteness, viewpoint, temporal_focus
      - topic: treat as the thematic anchor (generic/non-identifying)
      - seed: use only as a silent variation nudge for pacing, syntax, imagery, and vocabulary (never mention it)

   - Do NOT echo field names or values from the controls JSON.

    Output discipline
    - Return only what the caller’s external schema requires; no extra commentary, prefaces, or formatting outside the paragraph content."""

# Define axes to vary across:
CORE = {
    "domain": ["life_sciences","physical_sciences","engineering","computing",
               "humanities","social_sciences","everyday_scenarios","nature_travel","arts_culture"],
    "mode": ["narrative","expository","descriptive","process_explanation","persuasive"],
    "tone": ["plain","formal","technical","playful","reflective","conversational"],
    "reading_level": ["Grade8","Grade12","Undergraduate","Graduate"]
}

KNOBS = {
    "sentence_length": ["short","mixed","long"],
    "figurative_language": ["none","low","medium","high"],
    "concreteness": ["abstract","mixed","concrete"],
    "viewpoint": ["1st","2nd","3rd"],
    "temporal_focus": ["past","present","future"]
}

# Combine dictionaries. Leave one out if desired to create pairs on one or the other only.
FACTORS = {**CORE, **KNOBS}

names  = list(FACTORS.keys())
values = [FACTORS[k] for k in names]

rows = [dict(zip(names, combo)) for combo in AllPairs(values)]
print(f"Generated {len(rows)} rows for pairwise(t=2) coverage across CORE+KNOBS.")

pair_axes = list(itertools.combinations(names, 2))

universe = set()
for a, b in pair_axes:
    for va in FACTORS[a]:
        for vb in FACTORS[b]:
            universe.add((a, va, b, vb))

covered = set()
for r in rows:
    for a, b in pair_axes:
        covered.add((a, r[a], b, r[b]))

coverage_pct = 100.0 * len(covered) / len(universe)
print(f"Pairwise coverage: {len(covered)}/{len(universe)} = {coverage_pct:.2f}%")
assert covered == universe, "Not all pairs are covered — check factor lists or constraints."

#%%
n = 0

for i, r in enumerate(rows, 1):
    while n <= 5:
        controls = {
            # CORE
            "domain": r["domain"],
            "mode": r["mode"],
            "tone": r["tone"],
            "reading_level": r["reading_level"],
            # KNOBS nested under 'style'
            "style": {
                "sentence_length": r["sentence_length"],
                "figurative_language": r["figurative_language"],
                "concreteness": r["concreteness"],
                "viewpoint": r["viewpoint"],
                "temporal_focus": r["temporal_focus"],
            }
        }

        with open("paragraph_output_schema__single.json", "r") as jsonFile:
            output_schema = json.load(jsonFile)

        resp = client.responses.create(
            model="gpt-4o",
            instructions=SYSTEM_PROMPT,  # your fixed system prompt
            input=json.dumps({"controls": controls}),
            text={
                "format": {"type": "json_schema", "name": "math_response", "schema": output_schema, "strict": True}
            }
        )

        obj = json.loads(resp.output_text)  # one paragraph result
        with open("paragraphs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(f"Generated {i}/{len(rows)}")
        time.sleep(0.1)

        n += 1
