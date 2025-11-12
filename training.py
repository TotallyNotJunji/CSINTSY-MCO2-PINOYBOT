import numpy as np
import pandas as pd #besure to download sklearn and pandas before this
import math 
import pickle
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

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

#This method counts how many n grams in filipino a word has
#return: number of filipino n-grams in a word
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
    #likelyhood its 
    fil_score = filipino_n_gram(token) + int(filipino_prefix(token))
    eng_score = english_n_gram(token) + int(english_suffix(token))
    return int(fil_score > eng_score)

def is_likely_proper_noun(token):
    token = str(token) if pd.notna(token) else ""
    if len(token) <= 2:
        return 0
    return int(token[0].isupper() and not token.isupper())

#This method checks if a word starts with a common filipino pre-fix
#return: true if 
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
    return max(1, count) \


if __name__ == "__main__":
    data = pd.read_csv('final_annotations - final_annotations.csv')

  
    words = data['word']
    target = data['label']

    num_of_words = len(words)
    data['word_position_in_sentence'] = data.groupby('sentence_id').cumcount() + 1 
    num_of_features = 17 #CHANGE HERE IF YOU WANNA ADD FEATURES
    features = [[None for i in range(num_of_features)] for i in range(num_of_words)] 
    for i in range(0, num_of_words):
        features[i][0] = filipino_n_gram(words[i]) #how many ngrams in filipino does it have
        features[i][1] = english_n_gram(words[i]) #how many ngrams in eng does it have
        features[i][2] = vowel_to_letter_ratio(words[i]) #how many vowels to letters does it have 
        features[i][3] = len(str(words[i])) #length of words
        features[i][4] = letters_to_length_ratio(words[i]) #number of letters to word length ratio
        features[i][5] = uncommon_letters_in_filipino(words[i]) #num of uncommon letters in fil
        features[i][6] = has_digits(words[i]) #counts how many digits a word has 
        features[i][7] = capitalization_pattern(words[i]) #returns the capitalization pattern
        features[i][8] = filipino_prefix(words[i]) #returns if it has a prefix
        features[i][9] = is_likely_proper_noun(words[i]) 
        features[i][10] = mixed_script_pattern(words[i])
        features[i][11] = english_suffix(words[i])
        features[i][12] = filipino_suffix(words[i])
        features[i][13] = estimate_syllables(words[i])
        features[i][14] = data.loc[i, 'word_position_in_sentence']
        features[i][15] = is_internet_slang(words[i])
        features[i][16] = has_repeated_chars(words[i])

        #ADD ADDITIONAL FEATURES HERE

    X = np.array(features)
    y = np.array(target)

    words_array = np.array(words)
    
    X_train, X_temp, y_train, y_temp, words_train, words_temp = train_test_split(
        X, y, words_array, test_size=0.30
    )
    
    X_val, X_test, y_val, y_test, words_val, words_test = train_test_split(
        X_temp, y_temp, words_temp, test_size=0.50
    )
   
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        max_depth=15,          # Maximum depth of each tree
        min_samples_split=5    # Minimum samples to split a node
    )
    
    model.fit(X_train, y_train)
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    pred = model.predict(X_test)
    pred_val = model.predict(X_val)
   
    print("\n\n\nSAMPLE TRAINING PINOYBOT:")
    print("Sample predictions: ", pred)
    print("Actual sample: ", y_test)
    print('Number of training:', len(X_train))
    print('Number of tests: ', len(pred))
    print("\n=== VALIDATION SET ===")
    print("Sample predictions: ", pred_val)
    print("Actual sample: ", y_val)
    print('Number of training:', len(X_train))
    print('Number of validation: ', len(pred_val))
    print('Number of tests: ', len(pred))
    
    acc_val = accuracy_score(y_val, pred_val)
    print("\nValidation Accuracy:", acc_val)
    preci_val = precision_score(y_val, pred_val, average = None)
    print("Validation Precision: ", preci_val)
    f1_val = f1_score(y_val, pred_val, average=None)
    print("Validation F1 Score: ", f1_val)
    
    
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    preci = precision_score(y_test, pred, average = None)
    print("Precision: ", preci)
    f1 = f1_score(y_test, pred, average=None)
    print("F1 Score: ", f1)

    print("Mistakes:", len(pred) - (acc * len(pred)))
    print("Correct:", (acc * len(pred)))
    conf = confusion_matrix(y_test, pred)
    print()
    print("Confusion Matrix:")
    print(conf)

    unique_elements = list(set(y_test))
    print(unique_elements)

    print("\n=== Feature Importance ===")
    feature_names = [
        'filipino_ngram', 'english_ngram', 'vowel_ratio', 'length',
        'letters_ratio_non_alpha', 'uncommon_letters', 'has_digits', 'capitalization',
        'reduplication', 'filipino_prefix', 'proper_noun', 'mixed_script',
        'english_suffix', 'filipino_suffix', 'estimate_syllables', 'word_placement'
        ,'is_slang'
    ]

    importances = model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), 
    key=lambda x: x[1], reverse=True):
        print(f"{name}: {importance:.4f}")
        
    mismatch_indices = np.where(y_test != pred)[0]

    print("\n=== MISCLASSIFIED WORDS ===")
    print(f"Total misclassified: {len(mismatch_indices)}\n")

    max_show = 50  # show first 50 errors
    for i in mismatch_indices[:max_show]:
        word = words_test[i]
        actual = y_test[i]
        predicted = pred[i]
        print(f"Word: '{word}' | Actual: {actual} | Predicted: {predicted}")