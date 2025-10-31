import numpy as np
import pandas as pd #besure to download sklearn and pandas before this
import math
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

def uncomon_letters(token):
    if 'x' in token:
        return True
    if 'c' in token:
        return True
    if 'x' in token:
        return True
    return False

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


if __name__ == "__main__":
    data = pd.read_csv('final_annotations.csv')
    data = data.dropna(subset=['word'])
    data = data.reset_index(drop=True) #drop all the null words
  
    words = data['word']
 
    
    target = data['label']

    num_of_words = len(words)
    
    features = [[None for _ in range(6)] for _ in range(num_of_words)] 
    for i in range(0, num_of_words):
        features[i][0] = filipino_n_gram(words[i]) #how many ngrams in filipino does it have
        features[i][1] = english_n_gram(words[i]) #how many ngrams in eng does it have
        features[i][2] = vowel_to_letter_ratio(words[i]) #how many vowels to letters does it have 
        features[i][3] = len(words[i])
        features[i][4] = letters_to_length_ratio(words[i])
        features[i][5] = uncomon_letters(words[i])

    X = np.array(features)
    y = np.array(target)

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.30) #30% of the data will be used for testing 70% is for training
    
    model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n\n\nSAMPLE TRAINING PINOYBOT:")
    print("Sample predictions: ", pred)
    print("Actual sample: ", y_test)
    print('Number of training:', len(X_train))
    print('Number of tests: ', len(pred))
    
    
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    conf = confusion_matrix(y_test, pred)
    print()
    print("Confusion Matrix:")
    print(conf)


