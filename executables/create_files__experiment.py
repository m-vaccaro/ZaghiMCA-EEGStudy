import pandas as pd
import ast

# ---------- CONFIG ----------
# Input CSV (your "practice" database)
SRC_CSV = "../database_storage/database_19-gpt5_1-full-120_to_150_words__practice__embeddings-large__mcqs_3q.csv"

# Output CSV for PsychoPy practice loop
OUT_CSV = "../database_storage/final_GUI/test_trials.csv"

# Paragraph ID prefix for practice trials
PARA_PREFIX = "TEST"
# ----------------------------


def main():
    df = pd.read_csv(SRC_CSV)

    rows_out = []

    for i, row in df.iterrows():
        # Parse the 'choices' string into a real dict
        # e.g. "{'A': '...', 'B': '...', 'C': '...'}" -> {'A': '...', 'B': '...', 'C': '...'}
        choices = ast.literal_eval(row["choices"])

        # Pull choices into separate columns
        choiceA = choices.get("A", "")
        choiceB = choices.get("B", "")
        choiceC = choices.get("C", "")

        # Base question text
        question = str(row["question"]).strip()

        # Build one output row
        rows_out.append(
            {
                "trialIndex": i + 1,  # 1-based index
                "paragraphID": f"{PARA_PREFIX}{i + 1}",
                "paragraphText": row["text"],
                "compQText": question,
                "choiceA": f"a) {choiceA}",
                "choiceB": f"b) {choiceB}",
                "choiceC": f"c) {choiceC}",
                # Lowercase so it matches PsychoPy allowed keys ['a','b','c',...]
                "correctAns": str(row["correct_answer"]).lower(),
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} practice trials to {OUT_CSV}")


if __name__ == "__main__":
    main()
