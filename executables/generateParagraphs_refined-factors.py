import os
import json, random
from openai import OpenAI
import time
import pandas as pd
from testflows.combinatorics import CoveringArray

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))

# List of topics to generate texts for: life_sciences, physical_sciences, engineering, computing, humanities,
# social_sciences, everyday_scenarios, nature_travel, arts_culture

# Define messages
SYSTEM_PROMPT = """You generate exactly one output per request following the specified JSON schema.

    Hard Constraints
    - Produce a single paragraph only.
    - Make sure the paragraph is between 150 and 500 words. 
    - Write in English only.
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

canonical_names = list(FACTORS.keys())

# num_runs independent t=3 runs over FACTORS, then merge + deduplicate

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

    # build a t=3 covering array for this ordering
    params = {name: value_orders[name] for name in name_order}
    ca = CoveringArray(params, strength=3)
    assert ca.check()  # t=3 coverage

    # merge + dedupe immediately
    added = 0
    for r in ca:
        key = tuple((k, r[k]) for k in canonical_names)  # canonical key order
        if key not in seen:
            seen.add(key)
            rows_all.append(r)
            added += 1

    print(f"Run {run+1}/{num_runs}: added {added} unique rows (pool so far: {len(rows_all)})")

#%% Inspect coverage

# Tabular view of the design (CORE first, then KNOBS)
col_order = list(CORE.keys()) + list(KNOBS.keys())
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
with open("paragraph_output_schema__single.json", "r", encoding="utf-8") as jf:
    output_schema = json.load(jf)

# Add a method to handle exceptions thrown by improper JSON formatting
MAX_RETRIES = 4
BACKOFF_SEC = [0.5, 1, 2, 4]  # wait times of retries in case of errors associated with rate limits, etc.

duration_total = 0.0
N_total = len(rows_all)

global_start = time.time()

for i, r in enumerate(rows_all, 1):
    startTime = time.time()

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

    success = False
    raw = None  # for debug saves

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model="gpt-5.1",
                instructions=SYSTEM_PROMPT,
                input=json.dumps({"controls": controls}),
                reasoning={
                    "effort": "low"
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
            assert obj.get("domain") == controls["domain"], "Domain mismatch"
            assert isinstance(obj.get("text"), str) and len(obj["text"]) > 0, "Missing text"
            assert resp.incomplete_details is None

            # Save result
            with open("../paragraphs_gpt5_1.jsonl", "a", encoding="utf-8") as f:
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
df = pd.json_normalize(pd.read_json("../paragraphs_gpt5_1.jsonl", lines=True).to_dict(orient="records"))
df.to_csv("./database_storage/database_12-gpt5_1-full.csv", index=False)