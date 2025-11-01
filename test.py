import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# SAMPLE TESTING OF MACHINE LEARNING


if __name__ == "__main__":
    #sample data set, saving the sample into 'iris'
    iris = load_iris()
    #feature data set 2d array
    X = np.array(iris['data']) #data is a 2d array (n*4)

    #labels / correct answers
    y = np.array(iris['target'])

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.5)

    #TRAINING
    #constructor for the model, default is gini for descsion
    model = DecisionTreeClassifier()
    
    #trains the model, pass the feature and then the correct answer
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    #just some random stuff 
    print(pred)
    print(y_test)

    acc = confusion_matrix(y_test, pred)
    print(acc)

    
   