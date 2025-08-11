import os
import pandas as pd

from src.annotate.parse_results import parse_results
from src.annotate.prompts import get_sample_prompt

# -------------------------
# Small helper functions
# -------------------------

def _out_paths(save_dir: str, split_name: str, task: str, model_name: str):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{split_name}_{task}_{model_name}_results.csv")
    tmp_path = out_path + ".tmp"
    return out_path, tmp_path

def _load_existing(out_path: str) -> pd.DataFrame:
    cols = ['problem_num', 'A_offer', 'B_offer', 'model', 'result', 'task']
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        existing = df[cols]
        # drop na by 'result' to avoid empty rows
        existing = existing.dropna(subset=['result'], how='all')
        return existing
    return pd.DataFrame(columns=cols)

def _existing_key(existing_df: pd.DataFrame, task: str):
    """Return set of (problem_num, model, task) already present for this task."""
    subset = existing_df[existing_df['task'] == task]
    keys = set()
    for _, r in subset.iterrows():
        if pd.notna(r.problem_num) and pd.notna(r.model):
            keys.add((str(r.problem_num), str(r.model), str(task)))
    return keys

def _rows_to_process(df: pd.DataFrame, model_name: str, existing_keys, task: str):
    """Yield (idx, row) to annotate for this model+task."""
    for idx, row in df.iterrows():
        prob = str(row['problem_num'])
        if (prob, model_name, task) not in existing_keys:
            yield idx, row

def _annotate_one(row, model_name: str, task: str, get_response):
    """Return a dict row for CSV (handles parse + exceptions)."""
    A_offer = row['A']
    B_offer = row['B']
    problem_num = str(row['problem_num'])

    prompt = get_sample_prompt(A_offer, B_offer, task)
    try:
        raw = get_response(prompt)
        parsed = parse_results(raw, task)
        print(parsed)
    except Exception as e:
        print(f"[{model_name}] problem {problem_num}: {e}")
        parsed = None

    return {
        'problem_num': problem_num,
        'A_offer': A_offer,
        'B_offer': B_offer,
        'model': model_name,
        'result': parsed,
        'task': task
    }

def _append_and_atomic_save(existing_df, new_rows, out_path, tmp_path):
    """Append buffered rows; drop duplicates on (problem_num, model, task); atomic write."""
    if not new_rows:
        return existing_df  # nothing new
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    combined.sort_values(by=['problem_num', 'model', 'task'], inplace=True)
    combined = combined.drop_duplicates(
        subset=['problem_num', 'model', 'task'],
        keep='last'
    )

    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, out_path)
    print(f"Saved {len(combined)} rows → {out_path}")
    return combined

