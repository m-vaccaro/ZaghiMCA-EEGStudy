"""The goal of this script is to test for pairwise associations between variables in the dataset. More explicitly, the
aim is to test whether there is, for example, systematic confounding between any two style knobs (e.g., expository
paragraphs always tend to have difficulty=hard, etc.). If such an association is found, we cannot truly separate
these effects in the EEG analysis later."""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association

def cramers_v(x, y):
    cross_table = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(cross_table, correction=False) # Note: Set correction=False to avoid doing Yates' correction. This is not desirable for Cramer's V
    n_samples = cross_table.values.sum()

    # Calculate Cramer's V as SQRT((X^2/n_samples) / min(n_columns-1, n_rows-1))
    n_rows, n_cols = cross_table.shape

    min_dim_minus_one = min(n_rows,n_cols)-1

    v = np.sqrt((chi2 / n_samples) / min_dim_minus_one)

    v_association = association(cross_table.values, method="cramer", correction=False)

    assert v_association==v

    # Optionally apply Bergsma (2013)'s correction for Cramer's V
    phi2 = chi2 / n_samples
    phi2_corr = max(0, phi2 - ((n_cols - 1) * (n_rows - 1)) / (n_samples - 1))
    r_corr = n_rows - ((n_rows - 1) ** 2) / (n_samples - 1)
    c_corr = n_cols - ((n_cols - 1) ** 2) / (n_samples - 1)
    v_corr = np.sqrt(phi2_corr / min((c_corr - 1), (r_corr - 1)))

    print(v)
    print(v_corr)

    return v, v_corr, p, dof

if __name__ == "__main__":
    # 1. Load your CSV
    database_number = 19
    database_name = 'gpt5_1-full-120_to_150_words'
    df = pd.read_csv(f"../database_storage/database_{database_number}-{database_name}.csv")

    # 2. List the categorical factors you care about
    cat_cols = [
        "genre",
        "difficulty",
        "predictability",
        "emotional_valence",
        "concreteness",
        "tone",
        "topic_hint",
    ]

    # 3. Initialize empty matrices for V, V_corrected and p-values
    n = len(cat_cols)
    v_matrix = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)
    v_corr_matrix = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)
    p_matrix = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)
    dof_matrix = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)

    # 4. Compute Cramér's V for each pair
    for i, col_i in enumerate(cat_cols):
        for j, col_j in enumerate(cat_cols):
            v, v_corr, p, dof = cramers_v(df[col_i], df[col_j])
            v_matrix.loc[col_i, col_j] = v
            v_corr_matrix.loc[col_j, col_i] = v_corr
            p_matrix.loc[col_i, col_j] = p
            dof_matrix.loc[col_i, col_j] = dof

    # 5. Print results
    print("Cramér's V matrix:\n", v_matrix.round(3))
    print("\n\n\nChi-square p-value matrix:\n", p_matrix.round(3))
    print("\n\n\nCramér's V matrix (using Bergsma's correction):\n", v_corr_matrix.round(3))

    # 6. Save results to CSV
    v_matrix.to_csv(f"../database_storage/associations/cramers_v_database{database_number}.csv")
    v_corr_matrix.to_csv(f"../database_storage/associations/cramers_v_corr_database{database_number}.csv")
    p_matrix.to_csv(f"../database_storage/associations/cramers_v_pvalues_database{database_number}.csv")
    dof_matrix.to_csv(f"../database_storage/associations/cramers_v_dofs_database{database_number}.csv")

