"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""
#ONLY USE THIS FILE ONCE TRAINING.PY IS DONE!!!!
import os
import pickle
import pandas as pd
import numpy as np #numpy goat!

from typing import List


def is_internet_slang(token):
    token = str(token) if pd.notna(token) else ""
    slang = {'haha', 'hahaha', 'lol', 'omg', 'btw', 'hmu', 'pos', 'tas', 'pra'}
    return int(token.lower() in slang)

def has_repeated_chars(token):
    token = str(token) if pd.notna(token) else ""
    if len(token) < 3:
        return 0
    for i in range(len(token) - 2):
        if token[i] == token[i+1] == token[i+2]:
            return 1
    return 0

def filipino_n_gram(token):
    token = str(token).lower() if pd.notna(token) else ""
    count = 0
    if 'mag' in token:
        count = count + 1
    if 'ang' in token:
        count = count + 1
    if 'ala' in token:
        count = count + 1
    if 'isa' in token:
        count = count + 1
    if 'ng' in token:
        count = count + 1
    if 'nang' in token:
        count = count + 1
    if 'ksyon' in token: 
        count = count + 1
    return count

def filipino_suffix(token):
    token = str(token).lower() if pd.notna(token) else ""
    suffixes = ('an', 'in', 'ng', 'han', 'hin', 'ay')
    return int(token.endswith(suffixes))

def english_suffix(token):
    token = str(token).lower() if pd.notna(token) else ""
    suffixes = ('tion', 'ing', 'ed', 'ly', 'ment', 'ness', 'er', 'est')
    return int(token.endswith(suffixes))

def mixed_script_pattern(token):
    fil_score = filipino_n_gram(token) + int(filipino_prefix(token))
    eng_score = english_n_gram(token) + int(english_suffix(token))
    return int(fil_score > eng_score)

def is_likely_proper_noun(token):
    token = str(token) if pd.notna(token) else ""
    if len(token) <= 2:
        return 0
    return int(token[0].isupper() and not token.isupper())

def filipino_prefix(token):
    token = str(token).lower() if pd.notna(token) else ""
    prefixes = ("pagka", "pagkaka", "pina", "pinag", "ipa", "pa", "magpa", "nag", "mag", 'ka')
    return token.startswith(prefixes)

def english_n_gram(token):
    token = str(token).lower() if pd.notna(token) else ""
    count = 0
    if 'th' in token:
        count = count + 1
    if 'and' in token:
        count = count + 1
    if 'ion' in token:
        count = count + 1
    if 'ph' in token:
        count = count + 1
    return count

def uncommon_letters_in_filipino(token):
    token = str(token).lower() if pd.notna(token) else ""
    return token.count('x') + token.count('c') + token.count('z') + token.count('f') + token.count('q') + token.count('v') + token.count('j')

def letters_to_length_ratio(token):
    token = str(token) if pd.notna(token) else ""
    letters = 0
    for char in token:
        if not (char.isalpha()):
            letters = letters + 1
    if len(token) == 0:
        return 0
    return letters/len(token)

def vowel_to_letter_ratio(token):
    token = str(token) if pd.notna(token) else ""
    vowels = 'aeiouAEIOU'
    total = 0
    for char in token:
        if char.isalpha():
            total = total + 1
    vowel_count = 0
    
    for char in token:
        if char in vowels:
            vowel_count = vowel_count + 1
    if total == 0:
        return 0
    return vowel_count/total

def has_digits(token):
    token = str(token).lower() if pd.notna(token) else ""
    return any(char.isdigit() for char in token)

def capitalization_pattern(token):
    token = str(token) if pd.notna(token) else ""
    if len(token) == 0:
        return 0
    if token.isupper():
        return 2  # ALL CAPS
    if token[0].isupper():
        return 1  # first letter lang
    return 0  # lowercase

def estimate_syllables(token):
    token = str(token).lower() if pd.notna(token) else ""
    vowels = 'aeiou'
    count = 0
    prev_was_vowel = False
    for char in token:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    return max(1, count)

def extract_features(tokens: List[str]) -> np.ndarray:
    """
    Extract features from a list of tokens.
    
    Args:
        tokens: List of word tokens
    
    Returns:
        Feature matrix as numpy array (num_tokens x num_features)
    """
    num_of_features = 17
    num_tokens = len(tokens)
    features = [[None for _ in range(num_of_features)] for _ in range(num_tokens)]
    
    for i in range(num_tokens):
        word = tokens[i]
        # Position in sentence (1-indexed)
        word_position = i + 1
        
        features[i][0] = filipino_n_gram(word)
        features[i][1] = english_n_gram(word)
        features[i][2] = vowel_to_letter_ratio(word)
        features[i][3] = len(str(word))
        features[i][4] = letters_to_length_ratio(word)
        features[i][5] = uncommon_letters_in_filipino(word)
        features[i][6] = has_digits(word)
        features[i][7] = capitalization_pattern(word)
        features[i][8] = filipino_prefix(word)
        features[i][9] = is_likely_proper_noun(word)
        features[i][10] = mixed_script_pattern(word)
        features[i][11] = english_suffix(word)
        features[i][12] = filipino_suffix(word)
        features[i][13] = estimate_syllables(word)
        features[i][14] = word_position
        features[i][15] = is_internet_slang(word)
        features[i][16] = has_repeated_chars(word)
    
    return np.array(features)


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    
    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    #    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
    #    (Replace with your actual model loading code)
    MODEL_PATH = 'trained_model.pkl'
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Did not work KILL ME NOW!!!!")
        model = None

    # 2. Extract features from the input tokens to create the feature matrix
    #    Example: features = ... (your feature extraction logic here)
    features = extract_features(tokens)
    
    # 3. Use the model to predict the tags for each token
    #    Example: predicted = model.predict(features)
    predictions = model.predict(features)
    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]
    tags = [str(tag) for tag in predictions]
    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.
    return tags

if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print('Tags: ', tags)
