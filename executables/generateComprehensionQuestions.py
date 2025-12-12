#%%
"""The aim of this script is to generate comprehension questions based on each paragraph and, once generated, evaluate
the accuracy of each comprehension question using a separate GPT model instance. The goal is not to create overly
complex questions with difficult answers; rather, the purpose is only to check if readers are engaging with the material
or not. We want these questions to serve as a proxy for 'reader attention' during the experiment."""

import os
import json
import random

from openai import OpenAI
import time
import pandas as pd

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MCA"))
gpt_model = "gpt-5.1-2025-11-13"
reasoning_effort = "medium"   # Note: Reasoning effort is not applicable to some model generations.

database_number = 19
database_name = "gpt5_1-full-120_to_150_words__embeddings-large"

db = pd.read_csv(f"../database_storage/database_{database_number}-{database_name}.csv")
db_text = db["text"].tolist()

poss_answers = ["A", "B", "C", "D"]

# Define System and User messages
# The SYSTEM_PROMPT should:
#   Explicitly define the models role and the task assigned to it (e.g., create comprehension questions based on the given text only--requiring no outside knowledge and to generate the correct answer/explain the answer)
#   Question difficulty should be easy. Ideally the reader should be able to answer the question from just one read through the text.
#   Be explicit that the question is about the paragraph CONTENT, not the tone/etc. Again, readers must be able to answer the question with only the information given.
#   Still target core understanding of the text. Don't want trivial surface-level questions like "What was the main character's name?"
#   Require that one of A/B/C/D be correct (no all of the above)
#   Make sure questions are not worded as double negatives and that they are not opinion-based questions
#   Be designed to reduce guessing from participants/other cues in answers (e.g., all four options should be similar in length and style; the correct answer shouldn't be the only one that is specific, etc.)

SYSTEM_PROMPT = f"""You are an assistant that writes multiple-choice reading comprehension questions for short paragraphs.

YOUR ROLE AND TASK
- You are given exactly one paragraph of text.
- Your job is to write one multiple-choice question that tests understanding of that paragraph.
- The question you generate must be answerable using ONLY the information in the paragraph. Those answering these questions should never need to consult outside information to determine the correct answer.
- You must also generate the four answer options, identify which option is correct, and give a brief explanation for why each answer choice is right or wrong.

QUESTION DIFFICULTY AND FOCUS
- Aim for EASY difficulty. A careful reader should be able to answer correctly after reading through the paragraph once.
- The question must target core understanding of the paragraph (e.g., main idea, key fact, or causal link).
- Avoid trivial surface-level questions such as asking only for a name, a single number, or an isolated word unless that detail is central to understanding.
- The question must be about the CONTENT of the paragraph, not its tone, style, difficulty, or other writing characteristics.
- Do NOT ask about opinions, personal judgments, or anything that cannot be clearly determined from the paragraph.
- There must only be one correct answer.

ANSWER CHOICES (A–D)
- You must provide exactly FOUR answer options, labeled A, B, C, and D in the JSON object.
- Exactly ONE of A/B/C/D must be correct. The other three must be clearly incorrect for someone who has understood the paragraph.
- Do NOT use “all of the above”, “none of the above”, or similar meta-options.
- Write the options so that all four answer choices are similar in length, similar in level of detail, and similar in style and tone.
    - The correct option should NOT stand out by being much longer, more specific, more hedged, or obviously different from the other options.
- Avoid yes/no questions. Write questions with four contentful answer options instead.
- The answer choice label is embedded in the JSON schema. Do not add "A.", "B.", "C.", or "D." to the answer choices you generate.
- The user will specify what the correct answer should be in their message to you.

WORDING AND CLARITY
- Do not use double negatives or confusing logical structures.
- Do not start your questions with "According to the paragraph" or other similar phrasing. Each question will be paired directly with the paragraph you are given.
- The question and options must be clear, unambiguous, and grammatically correct.
- The question must have exactly one best answer based on the paragraph.
- A reader should be reasonably expected to get the correct answer after reading through the paragraph once without having to consult the text a second time.

OUTPUT FORMAT (STRICT)
- Your entire output MUST be a single JSON object following the given structure. 
- Always output a single JSON object as your entire response.
- The JSON object must have these fields:
  - "question": string — the question text.
  - "choices": object — with exactly four keys: "A", "B", "C", and "D". Each value is the answer text (without letter prefixes).
  - "correct_answer": string — exactly one of "A", "B", "C", or "D".
  - "explanation": string — 1–3 sentences explaining why the correct option is right and why the others are wrong, based only on the paragraph.
- Do NOT include any text outside the JSON object.
- Do NOT wrap the JSON in backticks or markdown.
- Do NOT add comments or extra fields."""

# Many of the instructions are included in the SYSTEM_PROMPT. Keep the USER_MESSAGE simple.
UMlist = [] # Store all USER_MESSAGE's for inspection to make sure nothing goes wrong (or to evaluate why something goes wrong)

# Load output schema
with open("../executables/output_schema__comprehension.json", "r", encoding="utf-8") as jf:
    output_schema = json.load(jf)

# Add a method to handle exceptions thrown by improper JSON formatting
MAX_RETRIES = 4
BACKOFF_SEC = [0.5, 1, 2, 4]  # wait times of retries in case of errors associated with rate limits, etc.

duration_total = 0.0
global_start = time.time()

n_texts = len(db_text)

for i in range(n_texts):
    startTime = time.time()
    USER_MESSAGE = f"""
Here is a paragraph:

{db_text[i].strip()}

Write one easy reading comprehension question about this paragraph that can be answered using only the information it contains. Make sure the correct answer is {random.choice(poss_answers)}."""
    UMlist.append(USER_MESSAGE)

    success = False
    raw = None  # for debug saves

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model="gpt-5.1-2025-11-13",
                instructions=SYSTEM_PROMPT,
                input=USER_MESSAGE,
                reasoning={
                    "effort": reasoning_effort
                },
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "question_generation",
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
            assert resp.incomplete_details is None

            # Save result
            if i == 0:
                mthd = "w"
            else:
                mthd = "a"

            with open(f"../database_storage/mcqs_database_{database_number}-{database_name}.jsonl", mthd,
                      encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            success = True
            break  # exit retry loop

        except json.JSONDecodeError as e:
            dbg_path = f"debug_response_{i}_try{attempt+1}.txt"
            try:
                with open(dbg_path, "w", encoding="utf-8") as dbg:
                    dbg.write(raw if raw is not None else "<no raw text available>")
                print(f"[warn] JSON parse failed at item {i+1} (try {attempt+1}): {e}. Saved {dbg_path}")
            except Exception:
                print(f"[warn] JSON parse failed at item {i+1} (try {attempt+1}): {e}. (Could not save debug file.)")
            time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC)-1)])

        except Exception as e:
            # Network/timeout/rate-limit/etc.
            print(f"[warn] API error at item {i+1} (try {attempt+1}): {e}")
            time.sleep(BACKOFF_SEC[min(attempt, len(BACKOFF_SEC)-1)])

    endTime = time.time()
    duration = endTime - startTime
    duration_total += duration
    duration_average = duration_total / (i + 1)
    numRemaining = n_texts - (i + 1)
    tRemaining = (numRemaining * duration_average) / 60
    tTotal = (time.time() - global_start) / 60

    if success:
        print(f"Generated {i+1}/{n_texts} in {duration:.2f}s. // Approx. {tRemaining:.2f} min remaining. // Total time: {tTotal:.2f} min.")
    else:
        print(f"[skip] Item {i+1} skipped after {MAX_RETRIES} attempts. Approx. {tRemaining:.2f} min remaining.")

    time.sleep(0.1)  # tiny pause to be gentle on rate limits

# Load JSONL and save to CSV
mcqs = pd.read_json(f"../database_storage/mcqs_database_{database_number}-{database_name}.jsonl", lines=True)
existing = pd.read_csv(f"../database_storage/database_{database_number}-{database_name}.csv")
merged = pd.concat([existing, mcqs], axis=1)
merged.to_csv(f"../database_storage/database_{database_number}-{database_name}__mcqs.csv", index=False)