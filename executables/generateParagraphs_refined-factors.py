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
SYSTEM_PROMPT = """You generate single-paragraph texts and a small amount of metadata following the given JSON format.

HARD CONSTRAINTS:
- Always output a single JSON object as your entire response.
- The JSON must have these fields: text, genre, difficulty, coherence_predictability, emotional_valence, concreteness, topic_hint, tone, keywords.
- The value of "text" must be exactly one paragraph of continuous prose (no headings, no bullet points, no dialogue formatting).
- Length of "text": between 250 and 400 words.
- Do NOT mention instructions, labels, field names, or any metadata in the paragraph.
- The paragraph must be self-contained and understandable on its own.

GENERAL CONTEXT:
- Paragraphs should involve STEM or academic themes (science, technology, engineering, mathematics, study, research, or student life), guided by the topic_hint the user provides.

FIELD INTERPRETATION:
Interpret each field of the input as follows:
- "genre":
  - "narrative": a short story or scene with at least one character and some change, decision, or outcome.
  - "expository": an informative explanation, description, or analysis of a concept, process, or situation.

- "difficulty":
  - "low": similar to non-fiction written for roughly middle school grades; short, simple sentences, common vocabulary, simple ideas, minimal jargon.
  - "medium": similar to non-fiction written for late high school or early college; moderate sentence length, some technical or abstract terms, concepts that require some effort.
  - "high": similar to advanced undergraduate or graduate-level material; longer and more complex sentences, dense information, frequent technical or abstract terminology.

- "coherence_predictability":
  - "high_coherence_high_predictability": the paragraph flows logically and the main point or ending feels natural and expected.
  - "high_coherence_low_predictability": the paragraph is logically coherent, but the final implication or twist is somewhat surprising while still plausible.
  - "low_coherence": all sentences are grammatically correct and broadly on the same general topic, but the flow is hard to follow (jumps in logic or missing links). Avoid obvious nonsense.

- "emotional_valence":
  - "negative": difficulties, setbacks, worry, or frustration.
  - "neutral": detached, matter-of-fact tone with little emotion.
  - "positive": curiosity, progress, satisfaction, or hope.

- "concreteness":
  - "abstract": mostly ideas, theories, principles, and general patterns, with few sensory details.
  - "mixed": a blend of abstract ideas and concrete examples.
  - "concrete": many specific objects, actions, places, or sensory details.

- "tone":
  - "plain": neutral, textbook-like, clear and straightforward.
  - "technical": more formal and jargon-heavy, closer to a scientific article or advanced textbook.
  - "reflective": slightly introspective or thoughtful, discussing experiences, challenges, or implications.

- "topic_hint":
  - Use the given topic area as a loose guide to the subject matter.

GENERAL BEHAVIOR:
- Treat the user-provided values for genre, difficulty, coherence_predictability, emotional_valence, concreteness, tone, and topic_hint as hard requirements.
- Do not explain your choices; just produce the JSON object."""

# Define axes to vary across:
FACTORS = {
    "genre": ["narrative", "expository"],
    "difficulty": ["low", "medium", "high"],
    "coherence_predictability": ["high_coherence_high_predictability", "high_coherence_low_predictability", "low_coherence"],
    "emotional_valence": ["negative", "neutral", "positive"],
    "concreteness": ["abstract", "mixed", "concrete"],
    "tone": ["plain", "technical", "reflective"]
}

TOPIC_HINTS = ["life_sciences",
               "physical_sciences",
               "engineering",
               "computing"]

canonical_names = list(FACTORS.keys())

# num_runs independent t=6 runs over FACTORS, then merge + deduplicate
# Note: If t = num. factors, then the space necessarily covers all possible combinations.

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

    # build a t=6 covering array for this ordering
    params = {name: value_orders[name] for name in name_order}
    ca = CoveringArray(params, strength=6)
    assert ca.check()  # t=6 coverage

    # merge + dedupe immediately
    added = 0
    for r in ca:
        key = tuple((k, r[k]) for k in canonical_names)  # canonical key order
        if key not in seen:
            seen.add(key)
            rows_all.append(r)
            added += 1

    print(f"Run {run+1}/{num_runs}: added {added} unique rows (pool so far: {len(rows_all)})")

for idx, row in enumerate(rows_all):
    row["topic_hint"] = TOPIC_HINTS[idx % len(TOPIC_HINTS)]

#%% Inspect coverage

# Tabular view of the design
col_order = list(FACTORS.keys()) + ["topic_hint"]
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
with open("executables/paragraph_output_schema__single_refined-factors.json", "r", encoding="utf-8") as jf:
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
        "genre": r["genre"],
        "difficulty": r["difficulty"],
        "coherence_predictability": r["coherence_predictability"],
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
                model="gpt-5.1",
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
            # (optional) sanity checks
            assert obj.get("genre") == controls["genre"], "Genre mismatch"
            assert obj.get("difficulty") == controls["difficulty"], "Difficulty mismatch"
            assert obj.get("coherence_predictability") == controls["coherence_predictability"], "Coherence/predictability mismatch"
            assert obj.get("emotional_valence") == controls["emotional_valence"], "Emotional valence mismatch"
            assert obj.get("concreteness") == controls["concreteness"], "Concreteness mismatch"
            assert obj.get("tone") == controls["tone"], "Tone mismatch"
            assert obj.get("topic_hint") == controls["topic_hint"], "Topic hint mismatch"
            assert isinstance(obj.get("text"), str) and len(obj["text"]) > 0, "Missing text"
            assert resp.incomplete_details is None

            # Save result
            with open("database_storage/paragraphs_gpt5_1.jsonl", "a", encoding="utf-8") as f:
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
df = pd.json_normalize(pd.read_json("database_storage/paragraphs_gpt5_1.jsonl", lines=True).to_dict(orient="records"))
df.to_csv("database_storage/database_13-gpt5_1-full.csv", index=False)