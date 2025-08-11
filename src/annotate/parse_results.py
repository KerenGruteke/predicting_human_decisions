import re

def parse_results(result, task):
    if task == "prob_and_value_for_winning":

        # use regex to find the tuple in the result
        match = re.search(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*([\d.]+)\s*\)', result)
        if match:
            value_A = int(match.group(1))
            value_B = int(match.group(2))
            probability_A_better = float(match.group(3))
            return (value_A, value_B, probability_A_better)
        else:
            raise ValueError(f"Result format is incorrect: {result}")
    
    elif "choose_offer_by_personality" in task:
        # Assuming result is a string in the format "(chosen_offer, value_A, value_B)"
        # use regex to find the tuple in the result
        match = re.search(r'\(\s*(\w)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', result)

        if match:
            chosen_offer = match.group(1)
            value_A = int(match.group(2))
            value_B = int(match.group(3))
            return (chosen_offer, value_A, value_B)
        else:
            raise ValueError(f"Result format is incorrect: {result}")
    
    else:
        raise ValueError(f"Unknown task: {task}")
    
    
if __name__ == "__main__":
    # Example usage
    print(parse_results("(50, 70, 0.6)", "prob_and_value_for_winning"))
    print(parse_results("(A, 50, 70)", "choose_offer_by_personality_Imagination-Based Thinker"))
    print(parse_results("(B, 30, 90)", "choose_offer_by_personality_Analytical Thinker"))
    # test also negative values
    print(parse_results("(-50, -70, 0.4)", "prob_and_value_for_winning"))
    print(parse_results("(A, -50, -70)", "choose_offer_by_personality_Imagination-Based Thinker"))
    print(parse_results("(B, -30, -90)", "choose_offer_by_personality_Analytical Thinker"))
    
    # example with spaces
    print(parse_results("( -20, 30, 0.1 )", "prob_and_value_for_winning"))
    print(parse_results("( -20 ,   30, 0.1 )", "prob_and_value_for_winning"))
    print(parse_results("( A , 20 , 30 )", "choose_offer_by_personality_Imagination-Based Thinker"))
    print(parse_results("( B , 20 , 30 )", "choose_offer_by_personality_Analytical Thinker"))