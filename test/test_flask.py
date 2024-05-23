from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate_model(X, y):
    model = KerasClassifier(build_fn=create_model, input_dim=X.shape[1], epochs=150, batch_size=10, verbose=0)
    scoring = ['precision', 'recall']
    results = cross_val_score(model, X, y, cv=5, scoring=scoring)
    return results
