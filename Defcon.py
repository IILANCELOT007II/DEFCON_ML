

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:,10].values
'''df = pd.read_csv('test.csv')
X_test = df.iloc[:, 0:10].values'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

'''df = pd.read_csv('test.csv')
X_test = dataset.iloc[:, 0:10].values'''

# Feature Scaling (THE MOST IMPORTANT PART IN DL)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


from sklearn.model_selection import GridSearchCV
parameters = {'criterion':['gini','entropy'],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'f1_weighted',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Predicting the Test set results
y_pred = grid_search.predict(X_test)

f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score: %f' % f1)
