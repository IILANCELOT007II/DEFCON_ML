

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset = dataset.drop(['Name'], axis = 1)
dataset = dataset.drop(['Ticket'], axis = 1)
dataset = dataset.drop(['Cabin'], axis = 1)
X = dataset.iloc[:, 2:10].values
y = dataset.iloc[:, 1].values
dataset = dataset[dataset['Embarked'].notna()]


from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 2:3])
X[:, 2:3]=missingvalues.transform(X[:, 2:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1= LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2= LabelEncoder()
X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])
onehotencoder = OneHotEncoder(categories=[6], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [6])],remainder='passthrough')
X = ct.fit_transform(X)

X = X[:, 1:]


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
          'n_estimators':range(10,50),
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}
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

