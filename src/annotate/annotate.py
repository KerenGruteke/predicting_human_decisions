import os
import json
import pandas as pd
import time
import google.generativeai as genai
from groq import Groq
from tqdm import tqdm
from src.annotate.utils import _out_paths, _load_existing, _existing_key, _rows_to_process, _annotate_one, _append_and_atomic_save

# Set up API clients
def read_api_key(file_path):
    with open(file_path) as f:
        return f.read().strip()

GROQ_KEY_PATH = os.path.join('API_keys', 'groq_key.txt')
GROQ_API_KEY = read_api_key(GROQ_KEY_PATH)
groq_client = Groq(api_key=GROQ_API_KEY)

GEMINI_KEY_PATH = os.path.join('API_keys', 'gemini_api_key.txt')
GEMINI_API_KEY = read_api_key(GEMINI_KEY_PATH)
genai.configure(api_key=GEMINI_API_KEY)

def get_llama_response(sample_prompt, model_name='llama3-70b-8192'):
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Please read the instructions carefully."},
            {"role": "user", "content": sample_prompt}
        ],
        temperature=0,
        max_tokens=2000
    )
    return response.choices[0].message.content

def get_gemini_response(sample_prompt, model_name='gemini-2.5-flash-lite'):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(sample_prompt)
    return response.text

# Main processing function

def annotate_df_by_task(
    df: pd.DataFrame,
    split_name: str,
    task: str,
    save_dir: str = "src/annotate",
    sleep_sec: int = 2,
    flush_every: int = 5
):
    """
    Annotate (problem_num, model, task) with LLM outputs.
    - Resumes: loads existing {split_name}_{task}_results.csv and skips already-done rows.
    - Writes atomically, flushes periodically.
    """

    buffered = []
    
    models = [
        ('llama', lambda x: get_llama_response(x)),
        ('gemini', lambda x: get_gemini_response(x)),
    ]
    model_names = [name for name, _ in models]
    print(model_names)

    for model_name, get_response in models:
        out_path, tmp_path = _out_paths(save_dir, split_name, task, model_name)
        existing = _load_existing(out_path)
        keys = _existing_key(existing, task)
        
        # decide what to annotate
        todo = list(_rows_to_process(df, model_name, keys, task))
        if not todo:
            print(f"[{model_name}] nothing to do — all rows already annotated for '{task}'.")
            continue

        print(f"[{model_name}] annotating {len(todo)} missing rows for task '{task}'...")
        for idx, row in tqdm(todo, desc=f"Processing {model_name}", unit="ex"):
            rec = _annotate_one(row, model_name, task, get_response)
            buffered.append(rec)

            # periodic flush
            if len(buffered) >= flush_every:
                existing = _append_and_atomic_save(existing, buffered, out_path, tmp_path)
                keys = _existing_key(existing, task)  # refresh
                buffered = []

            time.sleep(sleep_sec)

    # final flush
    _append_and_atomic_save(existing, buffered, out_path, tmp_path)


def annotate_dfs(task):
    train_test_file = "src/data/text_task/train_problems.csv"
    final_test_file = "src/data/text_task/test_problems.csv"
    
    for file_path in [train_test_file, final_test_file]:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue
        
        df = pd.read_csv(file_path)
        split_name = os.path.splitext(os.path.basename(file_path))[0]
        
        print(f"Processing {split_name} data...")
        annotate_df_by_task(df, split_name, task)
        print(f"Finished processing {split_name} data.")

if __name__ == "__main__":


    annotate_dfs(task="prob_and_value_for_winning")
    
    task = "choose_offer_by_personality"
    personalities_list = [
        "Risk-Averse", "Thrill-Seeking", "The Minimizer", "Expected Value Calculator",
        "Sure-Thing Lover", "The Pessimist", "Framing Follower", "Regret Minimizer",
        "Fairness Fanatic", "Availability Thinker", "Extremes Attractor", "Loss Averse",
        "Authority Believer", "Default Chooser", "Pattern Seeker", "Emotional Reactor",
        "Recency Thinker", "Word-Count Heuristic User", "Imagination-Based Thinker",
        "Random Chooser"
    ]
    # for personality in personalities_list:
    #     task_with_personality = f"choose_offer_by_personality_{personality}"
    #     print(f"Processing task: {task_with_personality}")
    #     annotate_dfs(task=task_with_personality)
