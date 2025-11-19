#%% Use this script to combine databases (for sets of texts generated in parts)

import pandas as pd
from pathlib import Path

fPaths = [Path("../databases_generated_by_parts") / f"database_07__SetBy75-Part{i + 1}.csv" for i in range(9)]

dfs = [pd.read_csv(p) for p in fPaths]
large_dataset = pd.concat(dfs, ignore_index=True)

#%% Save output to a .csv
database_name = "database_07__combined"
large_dataset.to_csv(f"{database_name}.csv", index=False)