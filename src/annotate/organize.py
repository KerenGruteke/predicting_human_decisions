import os
import re
import pandas as pd

# ---------- helpers ----------

def _safe_ratio(a: float, b: float):
    try:
        return float(a) / float(b) if float(b) != 0 else None
    except Exception:
        return None
    
def _calc_difference(a: float, b: float):
    """Calculate the difference between two values, handling None."""
    if a is None or b is None:
        return None
    return float(a) - float(b)

def _shorten_task_name(task: str) -> str:
    """Compact column-safe base for a task (keeps personality suffix if present)."""
    if "choose_offer_by_personality" in task:
        return "CBP" + task.split('_')[-1].replace(" ", "_")
    elif task == "prob_and_value_for_winning":
        return "PVW"
    else:
        raise ValueError(f"Unknown task: {task}")

def _parse_result_row(row):
    """
    Parse a single annotation row:
    - row has columns: problem_num, A_offer, B_offer, model, result, task
    - returns dict with parsed columns (names include task + model)
    """
    task = str(row["task"])
    model = str(row["model"])
    problem_num = row["problem_num"]

    parsed_cols = {}
    if not isinstance(row["result"], str) or row["result"].strip() == "":
        return {"problem_num": problem_num, **parsed_cols}  # nothing to parse

    # use your existing parse_results
    from parse_results import parse_results
    try:
        parsed = parse_results(row["result"], task)
    except Exception:
        return {"problem_num": problem_num}  # un-parseable

    short_task_name = _shorten_task_name(task)
    msuf = model  # suffix by model so multiple models can coexist

    # map parsed outputs to named columns based on task
    if task == "prob_and_value_for_winning":
        value_A, value_B, prob_A_better = parsed
        parsed_cols[f"{short_task_name}__val_A__{msuf}"] = value_A
        parsed_cols[f"{short_task_name}__val_B__{msuf}"] = value_B
        parsed_cols[f"{short_task_name}__prob_A_better__{msuf}"] = prob_A_better
        parsed_cols[f"{short_task_name}__A_over_B__{msuf}"] = _safe_ratio(value_A, value_B)
        parsed_cols[f"{short_task_name}__diff_A_B__{msuf}"] = _calc_difference(value_A, value_B)

    elif "choose_offer_by_personality" in task:
        chosen_offer, value_A, value_B = parsed
        parsed_cols[f"{short_task_name}__chosen_offer__{msuf}"] = chosen_offer
        parsed_cols[f"{short_task_name}__val_A__{msuf}"] = value_A
        parsed_cols[f"{short_task_name}__val_B__{msuf}"] = value_B
        parsed_cols[f"{short_task_name}__A_over_B__{msuf}"] = _safe_ratio(value_A, value_B)
        parsed_cols[f"{short_task_name}__diff_A_B__{msuf}"] = _calc_difference(value_A, value_B)

    else:
        # If you add more tasks later, handle here
        pass

    return {"problem_num": problem_num, **parsed_cols}

def widen_annotations(ann_df: pd.DataFrame) -> pd.DataFrame:
    """
    From the long annotations df (one row per (problem_num, task, model)),
    parse & return a wide df: one row per problem_num with all annotation columns.
    """
    parsed_rows = []
    for _, r in ann_df.iterrows():
        parsed_rows.append(_parse_result_row(r))

    if not parsed_rows:
        return pd.DataFrame(columns=["problem_num"])

    wide = pd.DataFrame(parsed_rows)
    # collapse duplicates (e.g., multiple attempts); keep last non-null
    wide = (
        wide.groupby("problem_num", as_index=False)
            .agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else None)
    )
    return wide

def merge_annotations_with_problems(problems_df: pd.DataFrame, wide_ann: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join problems (cols: problem_num, A, B) with wide annotations on problem_num.
    """
    base_cols = ["problem_num", "A", "B"]
    missing = [c for c in base_cols if c not in problems_df.columns]
    if missing:
        raise ValueError(f"problems_df missing required columns: {missing}")
    merged = problems_df[base_cols].merge(wide_ann, on="problem_num", how="left")
    return merged

# ---------- main entry points ----------

def build_and_save_train_test_annotations(
    annotations_csv_paths,
    train_problems_csv: str,
    test_problems_csv: str,
    out_dir: str = "src/annotate/combined"
):
    """
    - annotations_csv_paths: list of CSV paths (e.g., your per-task results files)
      columns required: problem_num, A_offer, B_offer, model, result, task
    - train_problems_csv, test_problems_csv: original problems files with (problem_num, A, B)
    - writes:
        out_dir/train_problems_with_annotations.csv
        out_dir/test_problems_with_annotations.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) load all annotations into one DataFrame
    ann_parts = []
    for p in annotations_csv_paths:
        if os.path.exists(p):
            part = pd.read_csv(p)
            # normalize expected columns
            need = ["problem_num", "A_offer", "B_offer", "model", "result", "task"]
            for c in need:
                if c not in part.columns:
                    part[c] = None
            ann_parts.append(part[need])
    if not ann_parts:
        raise ValueError("No annotations found. Provide valid annotations_csv_paths.")

    ann_all = pd.concat(ann_parts, ignore_index=True)

    # 2) widen annotations to one row per problem_num
    wide_ann = widen_annotations(ann_all)

    # 3) load train/test problems and merge
    train_df = pd.read_csv(train_problems_csv)
    test_df = pd.read_csv(test_problems_csv)

    train_out = merge_annotations_with_problems(train_df, wide_ann)
    test_out  = merge_annotations_with_problems(test_df,  wide_ann)

    # 4) save
    train_path = os.path.join(out_dir, "train_problems_with_annotations.csv")
    test_path  = os.path.join(out_dir, "test_problems_with_annotations.csv")
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)
    print(f"Saved:\n  {train_path}\n  {test_path}")

def get_annotations_for(df_with_problems: pd.DataFrame, annotations_csv_paths):
    """
    Given an input df (with 'problem_num'), return a DF of annotations
    for those same problem_nums (one row per problem).
    """
    ids = set(df_with_problems["problem_num"].astype(str))

    # load + filter annotations for these problems
    ann_parts = []
    for p in annotations_csv_paths:
        if os.path.exists(p):
            part = pd.read_csv(p)
            if "problem_num" in part.columns:
                ann_parts.append(part[part["problem_num"].astype(str).isin(ids)])
    if not ann_parts:
        return pd.DataFrame(columns=["problem_num"])

    ann_all = pd.concat(ann_parts, ignore_index=True)
    wide_ann = widen_annotations(ann_all)
    return wide_ann


if __name__ == "__main__":
    # Paths to your per‑task annotation CSVs (can be one or many)
    ann_paths = [
        "src/annotate/train_problems_prob_and_value_for_winning_gemini_results.csv",
        "src/annotate/test_problems_prob_and_value_for_winning_gemini_results.csv",
        "src/annotate/train_problems_prob_and_value_for_winning_llama_results.csv",
        "src/annotate/test_problems_prob_and_value_for_winning_llama_results.csv",
        # add more task files as you generate them
    ]

    # Build the merged train/test with annotations
    build_and_save_train_test_annotations(
        annotations_csv_paths=ann_paths,
        train_problems_csv="src/data/text_task/train_problems.csv",
        test_problems_csv="src/data/text_task/test_problems.csv",
        out_dir="src/annotate/combined"
    )

    # Get annotations for a subset df (same problem_nums) without saving
    subset = pd.read_csv("src/data/text_task/train_problems.csv").head(10)
    ann_subset = get_annotations_for(subset, ann_paths)
    print(ann_subset.columns)
    print(ann_subset.head())
