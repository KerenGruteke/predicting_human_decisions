import pandas as pd

def get_sample_prompt(offer_A, offer_B, task):
    if task == "prob_and_value_for_winning":
        return f"""
You are an estimator. 
Given two offers, A and B, your task is to estimate the probability that Offer A is better than Offer B, and the expected values of both offers.

Offer A: {offer_A}
Offer B: {offer_B}

Please provide your answer in in the following format:
A tuple of (value_A, value_B, probability_A_better) 
where:
1. value_A is the expected value of Offer A - an int between -100 and 100.
2. value_B is the expected value of Offer B - an int between -100 and 100.
3. probability_A_better is the probability that Offer A is better than Offer B - a float between 0 and 1.

Please ensure your response is formatted correctly.
"""

    elif "choose_offer_by_personality" in task:
        personality = task.split('_')[-1]
        
        decision_persinailities = pd.read_csv("src/annotate/decision_personalities.csv")
        if personality not in decision_persinailities['Personality Name'].values:
            raise ValueError(f"Personality '{personality}' not found in decision personalities data.")
        
        personality_description = decision_persinailities[decision_persinailities['Personality Name'] == personality]['Description'].values[0]
        
        # concise prompt that says they are simple people guided by this personality
        return f"""
You are a simple person guided by the personality: {personality_description}.
Given two offers, A and B, your task is to choose the offer that you would prefer based on your personality.

Offer A: {offer_A}
Offer B: {offer_B}

Please provide your answer in the following format:
A tuple of (chosen_offer, value_A, value_B) 
where:
1. chosen_offer is either 'A' or 'B', indicating which offer you prefer.
2. value_A is the expected value of Offer A - an int between -100 and 100.
3. value_B is the expected value of Offer B - an int between -100 and 100.

Please ensure your response is formatted correctly.
Provide only final answer!
"""

    else:
        raise ValueError(f"Unknown task: {task}")
    
    
if __name__ == "__main__":
    # Example usage
    print(get_sample_prompt(50, 70, "prob_and_value_for_winning"))
    print(get_sample_prompt(50, 70, "choose_offer_by_personality_Imagination-Based Thinker"))
