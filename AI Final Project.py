# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 02:20:07 2019

@author: clair
"""

import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from keras.metrics import categorical_accuracy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data Preparation Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
start_time = datetime.datetime.now()

data = pd.read_csv(r"C:\Users\clair\Google Drive\0 - AI\Assignment 6 - Final Project\bank/bank.csv", sep=';')
data.info()
data.head()


X = data.iloc[:,0:16]   # [all rows, col from index 2 to the last one]
y = data.iloc[:,16].values.tolist()  # [all rows, col one only y， I use values.tolist() change dataframe to list

y=[1 if x =='yes' 
   else 0 for x in y]
X = pd.get_dummies(X)  # Convert categorical variable into dummy/indicator variables

# Splitting Dataset: Now let's split our data into training and testing datasets.
# create a training/testing set 80/20, split train and test
# from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  =train_test_split(X,y,test_size = 0.2)  

# minmax scale x values
# from sklearn.preprocessing import MinMaxScaler
# The MinMaxScaler is essentially shrinks the range such that the range is now between 0 and 1
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train.shape, len(Y_train), X_test.shape, len(Y_test)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Grid Search

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_model(optimizer):
    model = Sequential()
    model.add(Dense(150, activation = 'relu', input_shape=(51,)))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = build_model)
parameters = {'batch_size': [8, ],
              'epochs': [40, 60],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)


best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("best_parameters: ")
print(best_parameters)
print("\nbest_accuracy: ")
print(best_accuracy)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Fit Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = Sequential()
model.add(Dense(12, activation = 'relu', input_shape=(51,)))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#Compiling ANN
# Compiling classifier. Using adam optimizer. Using binary_crossentropy for loss function since classification is binary, i.e. only two classes 'Yes' or 'Not'.
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history= model.fit(X_train,Y_train,batch_size =16, epochs = 60)

score = model.evaluate(X_test, Y_test, verbose=0) 
score[1]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('test_loss:', test_loss)
print('test_acc:', test_acc)

#plot 
plt.plot(np.arange(0,len(history.history['loss'])), history.history['loss'])
plt.title("Loss")
plt.show()
plt.plot(np.arange(0,len(history.history['acc'])), history.history['acc'])
plt.title("Accuracy")
plt.show()


stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)