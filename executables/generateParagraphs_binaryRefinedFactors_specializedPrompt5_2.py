import os
import json, random
from openai import OpenAI
import time
import pandas as pd
from testflows.combinatorics import CoveringArray
import itertools
import random

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))

database_number = 21
database_name ="gpt5_2-full-120_to_150_words"

gpt_model = "gpt-5.2-2025-12-11"

# List of topics to generate texts for: life_sciences, physical_sciences, engineering, computing, humanities,
# social_sciences, everyday_scenarios, nature_travel, arts_culture

# Define messages
SYSTEM_PROMPT = f"""Generate a single flat JSON object with these fields: text, genre, difficulty, predictability, emotional_valence, concreteness, topic_hint, tone. Strictly follow these definitions and constraints:

- text: One logically coherent, self-contained paragraph (120–150 words) on the user-provided STEM topic_hint. No headings, lists, dialogue, metadata, or references to instructions. Do not use em-dashes.

'genre' will be specified as 'narrative' or 'expository':
- 'narrative': Story or scene with at least one character and an outcome or decision.
- 'expository': Factual explanation or analysis of a concept, process, or situation.

'difficulty' will be specified as 'low' or 'high':
- 'low': Middle school-level clarity, short/simple sentences, little jargon.
- 'high': Dense, advanced undergraduate/graduate-level content; longer sentences, technical terms.

'predictability' will be specified as 'low' or 'high':
- 'low': Main outcome or point is logical but somewhat non-obvious or surprising.
- 'high': Main outcome or point is expected based on the setup.

'emotional_valence' will be specified as 'negative', 'neutral', or 'positive':
- 'negative': Conveys mild academic/STEM setbacks or concerns; avoid extremes.
- 'neutral': Factual, analytic, uncolored tone.
- 'positive': Shows curiosity, progress, satisfaction, or hope in a STEM setting—mild/moderate positivity.

'concreteness' will be specified as 'abstract' or 'concrete' (emphasize specifics or sensory detail).
- 'abstract': Focus on ideas, theories, or principles. Focus on concepts rather than tangible objects or specific scenes.
- 'concrete': Emphasize specifics or sensory detail.

'tone' will be specified as 'plain' or 'reflective':
- 'plain': Provide neutral, textbook-like explanations. Be direct and unembellished.
- 'reflective': Be introspective, and discuss experiences, challenges, or implications.

'topic_hint' will be specified as 'life_sciences', 'physical_sciences', 'engineering', or 'computing': 
- Use the given topic area as a guide to the subject matter. The paragraph should clearly relate to this area, but you have flexibility in choosing the specific scenario or example.

ALWAYS enforce all definitions and field compatibility; if user's values conflict, prioritize these definitions. Output only the JSON object and nothing else. Do not include explanations, commentary, or instructions. The text must always satisfy the user's field values and these constraints."""

# Define axes to vary across:
FACTORS = {
    "genre": ["narrative", "expository"],
    "difficulty": ["low", "high"],
    "predictability": ["low", "high"],
    "emotional_valence": ["negative", "neutral", "positive"],
    "concreteness": ["abstract", "concrete"],
    "tone": ["plain", "reflective"],
    "topic_hint": ["life_sciences", "physical_sciences", "engineering", "computing"]
}

canonical_names = list(FACTORS.keys())

# num_runs independent t=ca_strength runs over FACTORS, then merge + deduplicate
# Note: If t = num. factors, then the space necessarily covers all possible combinations.

ca_strength = 3       # Defines a t = ca_strength covering array

num_runs = 2          # bump this to 10–20 if you want a bigger, richer pool
base_seed = 23498     # change to vary the sequence

rows_all = []
seen = set()

for run in range(num_runs):
    rng = random.Random(base_seed + run)

    # randomize factor order and each factor's level order
    name_order = canonical_names[:]
    rng.shuffle(name_order)
    value_orders = {k: FACTORS[k][:] for k in canonical_names}
    for k in canonical_names:
        rng.shuffle(value_orders[k])

    # build a t=ca_strength covering array for this ordering
    params = {name: value_orders[name] for name in name_order}
    ca = CoveringArray(params, strength=ca_strength)
    assert ca.check()  # t=ca_strength coverage

    # merge + dedupe immediately
    added = 0
    for r in ca:
        key = tuple((k, r[k]) for k in canonical_names)  # canonical key order
        if key not in seen:
            seen.add(key)
            rows_all.append(r)
            added += 1

    print(f"Run {run+1}/{num_runs}: added {added} unique rows (pool so far: {len(rows_all)})")

# === Add practice (warm-up) rows with random *unique* combinations ===
num_practice = 2 # The covering array generates 63 combinations with t=3, so add 2 to get total 65 paragraphs
practice_rows = []

rng_practice = random.Random(base_seed + 198)  # different seed from CA runs

# Pick the first practice row randomly:
while len(practice_rows) < 1:
    combo = {name: rng_practice.choice(FACTORS[name]) for name in canonical_names}
    key = tuple((k, combo[k]) for k in canonical_names)

    if key in seen:
        continue

    seen.add(key)
    practice_rows.append(combo)

first_practice = practice_rows[0]

# Now develop a second practice that is not in 'seen' and differs from 'frist_practice' on every factor:
candidates = []
for values in itertools.product(*[FACTORS[name] for name in canonical_names]):
    combo = dict(zip(canonical_names, values))
    key = tuple((k, combo[k]) for k in canonical_names)

    if key in seen:
        continue

    # requre that every factor differs from the first practice row
    if all(combo[name] != first_practice[name] for name in canonical_names):
        candidates.append(combo)

if not candidates:
    raise RuntimeError("No valid candidate found for a maximally different practice row.")

second_practice = rng_practice.choice(candidates)
key_second = tuple((k, second_practice[k]) for k in canonical_names)
seen.add(key_second)
practice_rows.append(second_practice)

print(f"\nGenerated {len(practice_rows)} practice rows (unique from main design).")
print(pd.DataFrame(practice_rows))


#%% Inspect coverage

# Tabular view of the design
col_order = list(FACTORS.keys())
df = pd.DataFrame(rows_all)[col_order]

# Save the matrix for offline inspection
df.to_csv("controls_matrix.csv", index=False)
print("\nSaved controls_matrix.csv")

# Per-factor level counts (helps spot imbalance)
print("\n=== Per-factor level counts ===")
for col in col_order:
    vc = df[col].value_counts().sort_index()
    print(f"\n[{col}]")
    print(vc.to_string())


#%%

# Load output schema
with open("../executables/paragraph_output_schema__single_binary-refined-factors.json", "r", encoding="utf-8") as jf:
    output_schema = json.load(jf)

# Add a method to handle exceptions thrown by improper JSON formatting
MAX_RETRIES = 4
BACKOFF_SEC = [0.5, 1, 2, 4]  # wait times of retries in case of errors associated with rate limits, etc.

duration_total = 0.0
N_total = num_practice

global_start = time.time()

# generate practice trials and save in a separate file
for i, r in enumerate(practice_rows, 1):
    startTime = time.time()
    controls = {
        "genre": r["genre"],
        "difficulty": r["difficulty"],
        "predictability": r["predictability"],
        "emotional_valence": r["emotional_valence"],
        "concreteness": r["concreteness"],
        "tone": r["tone"],
        "topic_hint": r["topic_hint"],
    }

    success = False
    raw = None  # for debug saves

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=gpt_model,
                instructions=SYSTEM_PROMPT,
                input=json.dumps({"controls": controls}),
                reasoning={
                    "effort": "medium"
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "eeg_paragraph",
                        "schema": output_schema,
                        "strict": True
                    }
                }
            )

            # ---- robust extraction across response shapes ----
            obj = None
            raw = getattr(resp, "output_text", None)

            # New Responses shape
            out_list = getattr(resp, "output", None)
            if obj is None and isinstance(out_list, list) and out_list:
                content = getattr(out_list[0], "content", None)
                if isinstance(content, list) and content:
                    frag = content[0]
                    parsed = getattr(frag, "parsed", None)
                    if parsed is not None:
                        obj = parsed
                    else:
                        raw = getattr(frag, "text", raw)

            # Legacy chat-like fallback
            if obj is None and (raw is None) and hasattr(resp, "choices"):
                msg = getattr(resp.choices[0], "message", None)
                if msg is not None:
                    parsed = getattr(msg, "parsed", None)
                    if parsed is not None:
                        obj = parsed
                    else:
                        raw = getattr(msg, "content", raw)

            # Final parse
            if obj is None:
                if raw is None:
                    raise ValueError("No JSON text available in response.")
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", "replace")
                obj = json.loads(raw)
            # ---- end extraction ----

            # (optional) sanity checks
            assert obj.get("genre") == controls["genre"], "Genre mismatch"
            assert obj.get("difficulty") == controls["difficulty"], "Difficulty mismatch"
            assert obj.get("predictability") == controls[
                "predictability"], "Predictability mismatch"
            assert obj.get("emotional_valence") == controls["emotional_valence"], "Emotional valence mismatch"
            assert obj.get("concreteness") == controls["concreteness"], "Concreteness mismatch"
            assert obj.get("tone") == controls["tone"], "Tone mismatch"
            assert obj.get("topic_hint") == controls["topic_hint"], "Topic hint mismatch"
            assert isinstance(obj.get("text"), str) and len(obj["text"]) > 0, "Missing text"
            assert resp.incomplete_details is None

            # Save result
            with open(f"../database_storage/paragraphs_{database_name}__practice.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            success = True
            break  # exit retry loop

        except json.JSONDecodeError as e:
            dbg_path = f"debug_response_{i}_try{attempt+1}.txt"
            try:
                with open(dbg_path, "w", encoding="utf-8") as dbg:
                    dbg.write(raw if raw is not None else "<no raw text available>")
                print(f"[warn] JSON parse failed at item {i} (try {attempt+1}): {e}. Saved {dbg_path}")
            except Exception:
                print(f"[warn] JSON parse failed at item {i} (try {attempt+1}): {e}. (Could not save debug file.)")
            time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC)-1)])

        except Exception as e:
            # Network/timeout/rate-limit/etc.
            print(f"[warn] API error at item {i} (try {attempt+1}): {e}")
            time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC)-1)])

    endTime = time.time()
    duration = endTime - startTime
    duration_total += duration
    duration_average = duration_total / i
    numRemaining = N_total - i
    tRemaining = (numRemaining * duration_average) / 60
    tTotal = (time.time() - global_start) / 60

    if success:
        print(f"Generated {i}/{N_total} in {duration:.2f}s. // Approx. {tRemaining:.2f} min remaining. // Total time: {tTotal:.2f} min.")
    else:
        print(f"[skip] Item {i} skipped after {MAX_RETRIES} attempts. Approx. {tRemaining:.2f} min remaining.")

    time.sleep(0.1)  # tiny pause to be gentle on rate limits

df = pd.json_normalize(pd.read_json(f"../database_storage/paragraphs_{database_name}__practice.jsonl", lines=True).to_dict(orient="records"))
df.to_csv(f"../database_storage/database_{database_number}-{database_name}__practice.csv", index=False)

#%% generate for main dataset

duration_total = 0.0
N_total = len(rows_all)

global_start = time.time()
for i, r in enumerate(rows_all, 1):
    startTime = time.time()

    controls = {
        # CORE
        "genre": r["genre"],
        "difficulty": r["difficulty"],
        "predictability": r["predictability"],
        "emotional_valence": r["emotional_valence"],
        "concreteness": r["concreteness"],
        "tone": r["tone"],
        "topic_hint": r["topic_hint"]
    }

    success = False
    raw = None  # for debug saves

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=gpt_model,
                instructions=SYSTEM_PROMPT,
                input=json.dumps({"controls": controls}),
                reasoning={
                    "effort": "medium"
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "eeg_paragraph",
                        "schema": output_schema,
                        "strict": True
                    }
                }
            )

            # ---- robust extraction across response shapes ----
            obj = None
            raw = getattr(resp, "output_text", None)

            # New Responses shape
            out_list = getattr(resp, "output", None)
            if obj is None and isinstance(out_list, list) and out_list:
                content = getattr(out_list[0], "content", None)
                if isinstance(content, list) and content:
                    frag = content[0]
                    parsed = getattr(frag, "parsed", None)
                    if parsed is not None:
                        obj = parsed
                    else:
                        raw = getattr(frag, "text", raw)

            # Legacy chat-like fallback
            if obj is None and (raw is None) and hasattr(resp, "choices"):
                msg = getattr(resp.choices[0], "message", None)
                if msg is not None:
                    parsed = getattr(msg, "parsed", None)
                    if parsed is not None:
                        obj = parsed
                    else:
                        raw = getattr(msg, "content", raw)

            # Final parse
            if obj is None:
                if raw is None:
                    raise ValueError("No JSON text available in response.")
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", "replace")
                obj = json.loads(raw)
            # ---- end extraction ----

            # (optional) sanity checks
            assert obj.get("genre") == controls["genre"], "Genre mismatch"
            assert obj.get("difficulty") == controls["difficulty"], "Difficulty mismatch"
            assert obj.get("predictability") == controls["predictability"], "Predictability mismatch"
            assert obj.get("emotional_valence") == controls["emotional_valence"], "Emotional valence mismatch"
            assert obj.get("concreteness") == controls["concreteness"], "Concreteness mismatch"
            assert obj.get("tone") == controls["tone"], "Tone mismatch"
            assert obj.get("topic_hint") == controls["topic_hint"], "Topic hint mismatch"
            assert isinstance(obj.get("text"), str) and len(obj["text"]) > 0, "Missing text"
            assert resp.incomplete_details is None

            # Save result
            with open(f"../database_storage/paragraphs_{database_name}", "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            success = True
            break  # exit retry loop

        except json.JSONDecodeError as e:
            dbg_path = f"debug_response_{i}_try{attempt+1}.txt"
            try:
                with open(dbg_path, "w", encoding="utf-8") as dbg:
                    dbg.write(raw if raw is not None else "<no raw text available>")
                print(f"[warn] JSON parse failed at item {i} (try {attempt+1}): {e}. Saved {dbg_path}")
            except Exception:
                print(f"[warn] JSON parse failed at item {i} (try {attempt+1}): {e}. (Could not save debug file.)")
            time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC)-1)])

        except Exception as e:
            # Network/timeout/rate-limit/etc.
            print(f"[warn] API error at item {i} (try {attempt+1}): {e}")
            time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC)-1)])

    endTime = time.time()
    duration = endTime - startTime
    duration_total += duration
    duration_average = duration_total / i
    numRemaining = N_total - i
    tRemaining = (numRemaining * duration_average) / 60
    tTotal = (time.time() - global_start) / 60

    if success:
        print(f"Generated {i}/{N_total} in {duration:.2f}s. // Approx. {tRemaining:.2f} min remaining. // Total time: {tTotal:.2f} min.")
    else:
        print(f"[skip] Item {i} skipped after {MAX_RETRIES} attempts. Approx. {tRemaining:.2f} min remaining.")

    time.sleep(0.1)  # tiny pause to be gentle on rate limits


#%% Load JSONL, flatten nested fields (e.g., style.*), and save to CSV
df = pd.json_normalize(pd.read_json(f"../database_storage/paragraphs_{database_name}", lines=True).to_dict(orient="records"))
df.to_csv(f"../database_storage/database_{database_number}-{database_name}.csv", index=False)