"""
Load in the agreeement attraction stimuli from the csv file and 
load it and loop get_verb_probabilities over every row to get the verb probabilities for each item.
"""

import pandas as pd
from model import get_verb_probabilities

def run_verb_probability_analysis(model, tokenizer, csv_file):

    stimuli_df = pd.read_csv(csv_file)
    
    results = []
    
    for index, row in stimuli_df.iterrows():
        prefix_baseline = row['prefix_baseline']
        prefix_attractor = row['prefix_attractor']
        verb_correct = row['verb_correct']
        verb_incorrect = row['verb_incorrect']

        # Get probabilities for the baseline condition
        baseline_probs = get_verb_probabilities(model, tokenizer, prefix_baseline, verb_correct, verb_incorrect)
        # Get probabilities for the attractor condition
        attractor_probs = get_verb_probabilities(model, tokenizer, prefix_attractor, verb_correct, verb_incorrect)  

        results.append({
            'item': row['item_id'],
            'baseline_correct_prob': baseline_probs[verb_correct],
            'baseline_incorrect_prob': baseline_probs[verb_incorrect],
            'attractor_correct_prob': attractor_probs[verb_correct],
            'attractor_incorrect_prob': attractor_probs[verb_incorrect]
        })
    
    return pd.DataFrame(results)