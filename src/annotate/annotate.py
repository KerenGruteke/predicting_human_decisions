import os
import json
import pandas as pd
import time
import google.generativeai as genai
from groq import Groq
from tqdm import tqdm
from src.annotate.parse_results import parse_results
from src.annotate.prompts import get_sample_prompt

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

def get_llama_response(sample_prompt, model_name='llama-3.1-70b-versatile'):
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

def get_gemini_response(sample_prompt, model_name='gemini-1.5-flash'):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(sample_prompt)
    return response.text

# Main processing function
def annotate_df_by_task(df, split_name, task):
    # Define model configurations
    models = [
        ('gemini', lambda x: get_gemini_response(x)),
        ('llama', lambda x: get_llama_response(x)),
    ]

    results = []
    
    for model_name, get_response in models:
        for index, row in tqdm(df.iterrows(), desc=f"Processing {model_name}"):
            A_offer = row['A']
            B_offer = row['B']
            problem_num = row['problem_num']
            
            sample_prompt = get_sample_prompt(A_offer, B_offer, task)
            
            try:
                result = get_response(sample_prompt)
                parsed_result = parse_results(result, task)
                print(parsed_result, "\n")

                results.append((problem_num, A_offer, B_offer, model_name, parsed_result))
            except Exception as e:
                print(f"Error processing row {index} with {model_name}: {e}")
                results.append((problem_num, A_offer, B_offer, model_name, None))

            # Add delay between API calls to respect rate limits
            time.sleep(5)
        
    # convert to DataFrame
    results_df = pd.DataFrame(results, columns=['problem_num', 'A_offer', 'B_offer', 'model', 'result'])

    # Save results to CSV
    results_df.to_csv(f'src/annotate/{split_name}_results.csv', index=False)

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
