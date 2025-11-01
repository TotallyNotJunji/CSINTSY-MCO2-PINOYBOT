import numpy as np
import pandas as pd #besure to download sklearn and pandas before this
import math
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def filipino_n_gram(token):
    token = str(token).lower()
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
    return count

def english_n_gram(token):
    token = str(token).lower()
    count = 0
    if 'the' in token:
        count = count + 1
    if 'and' in token:
        count = count + 1
    if 'tha' in token:
        count = count + 1
    if 'ion' in token:
        count = count + 1
    return count

def uncommon_letters_in_filipino(token):
    return token.count('x') + token.count('c') + token.count('z') + token.count('f') + token.count('q') + token.count('v') 

def letters_to_length_ratio(token):
    letters = 0
    for char in token:
        if not (char.isalpha()):
            letters = letters + 1
    if len(token) == 0:
        return 0
    return letters/len(token)

def vowel_to_letter_ratio(token):
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
    return any(char.isdigit() for char in token)

def capitalization_pattern(token):
    if len(token) == 0:
        return 0
    if token.isupper():
        return 2  # ALL CAPS
    elif token[0].isupper():
        return 1  # first letter lang
    else:
        return 0  # lowercase

def is_reducplication(token):
    if '-' not in token and ' ' not in token:
        return False
    token = str(token).lower()
    halves = token.split('-')
    if len(halves) == 2 and halves[0] == halves[1]:
        return True
    halves = token.split(' ')
    if len(halves) == 2 and halves[0] == halves[1]:
        return True
    return False

if __name__ == "__main__":
    data = pd.read_csv('final_annotations.csv')
    data = data.dropna(subset=['word'])
    data = data.reset_index(drop=True) #drop all the null words
  
    words = data['word']
    target = data['label']

    num_of_words = len(words)

    num_of_features = 9 #CHANGE HERE IF YOU WANNA ADD FEATURES
    features = [[None for i in range(num_of_features)] for i in range(num_of_words)] 
    for i in range(0, num_of_words):
        features[i][0] = filipino_n_gram(words[i]) #how many ngrams in filipino does it have
        features[i][1] = english_n_gram(words[i]) #how many ngrams in eng does it have
        features[i][2] = vowel_to_letter_ratio(words[i]) #how many vowels to letters does it have 
        features[i][3] = len(words[i]) #
        features[i][4] = letters_to_length_ratio(words[i])
        features[i][5] = uncommon_letters_in_filipino(words[i])
        features[i][6] = has_digits(words[i])
        features[i][7] = capitalization_pattern(words[i])
        features[i][8] = is_reducplication(words[i])
        #ADD ADDITIONAL FEATURES HERE

    X = np.array(features)
    y = np.array(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4) #30% of the data will be used for testing 70% is for training
    
    model = DecisionTreeClassifier(max_depth=10)
    
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n\n\nSAMPLE TRAINING PINOYBOT:")
    print("Sample predictions: ", pred)
    print("Actual sample: ", y_test)
    print('Number of training:', len(X_train))
    print('Number of tests: ', len(pred))
    
    
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    print("Mistakes:", len(pred) - (acc * len(pred)))
    print("Correct:", (acc * len(pred)))
    conf = confusion_matrix(y_test, pred)
    print()
    print("Confusion Matrix:")
    print(conf)

    unique_elements = list(set(y_test))
    print(unique_elements)

  
