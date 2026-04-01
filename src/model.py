"""
File that will load GPT-2 small via TransformerLens, and a function that takes a 
sentence prefix plus two verb choices and returns the probability of each.
"""

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

def load_gpt2_small():
    """
    Load the GPT-2 small model and tokenizer using TransformerLens.
    """
    model_name = "gpt2-small"
    model = HookedTransformer.from_pretrained(model_name)
    # TransformerLens model names (like gpt2-small) don't always match Hugging Face repo ids.
    # The tokenizer is usually already attached to the loaded model.
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

# basically given this sentence starter, how likely is the next word to be verb1 vs verb2?
def get_verb_probabilities(model, tokenizer, sentence_prefix, verb1, verb2):
    """
    Given a sentence prefix and two verb choices, return the probability of each verb
    being the next word in the sequence according to the model.
    
    Args:
        model: The loaded GPT-2 small model.
        tokenizer: The corresponding tokenizer for GPT-2 small.
        sentence_prefix: A string representing the beginning of a sentence.
        verb1: The first verb choice as a string.
        verb2: The second verb choice as a string.
    """
    # Tokenize the input sentence prefix (token IDs)
    input_ids = tokenizer.encode(sentence_prefix, return_tensors='pt')
    
    # Get the model's output logits for the next token.
    # Makes sure the input ids are on the same device as the model (e.g., CPU or GPU)
    device = getattr(model, "device", None) or getattr(getattr(model, "cfg", None), "device", None)
    if device is not None:
        input_ids = input_ids.to(device)
    # not training, so we don't need gradients just predicting.
    # This also speeds up inference and reduces memory usage.
    with torch.no_grad():
        outputs = model(input_ids)
        # Depending on the model's output structure, we may need to access the logits differently.
        # this says either way get the output logits, if the model's output has a 'logits' attribute, use that; otherwise, assume the output itself is the logits.
        all_logits = outputs.logits if hasattr(outputs, "logits") else outputs
        logits = all_logits[:, -1, :]  # scores for the next token
    
    # Get the token IDs for the two verb choices
    verb1_id = tokenizer.encode(" " + verb1, add_special_tokens=False)[0]
    verb2_id = tokenizer.encode(" " + verb2, add_special_tokens=False)[0]
    
    # Calculate probabilities using softmax
    # In simple terms, softmax takes the raw scores (logits) and converts them into probabilities that sum to 1.
    probabilities = torch.softmax(logits, dim=-1)
    
    # convert the probabilities for the two verbs to Python floats and return them in a dictionary
    verb1_prob = probabilities[0, verb1_id].item()
    verb2_prob = probabilities[0, verb2_id].item()
    
    return {verb1: verb1_prob, verb2: verb2_prob}

# Example usage
def main():
    model, tokenizer = load_gpt2_small()
    sentence_prefix = "The cat"
    verb1 = "sat"
    verb2 = "jumped"
        
    probabilities_cat = get_verb_probabilities(model, tokenizer, sentence_prefix, verb1, verb2)
    probabilities_singular = get_verb_probabilities(model, tokenizer, "The key to the cabinet", "is", "are")
    probabilities_plural = get_verb_probabilities(model, tokenizer, "The key to the cabinets", "is", "are")
    print("Singular:", probabilities_singular)
    print("Plural:", probabilities_plural)
    print(probabilities_cat)
    

if __name__ == "__main__":
    main()