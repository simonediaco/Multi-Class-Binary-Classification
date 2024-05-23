from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging

class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, epochs=150, batch_size=10):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.classes_ = None

    def build_model(self):
        logging.debug("Building model")
        model = Sequential()
        model.add(Dense(12, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def fit(self, X, y):
        logging.debug(f"Fitting model with data shape X: {X.shape}, y: {y.shape}")
        self.classes_ = np.unique(y)
        self.model = self.build_model()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict(self, X):
        logging.debug(f"Predicting with model, data shape X: {X.shape}")
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        logging.debug(f"Predicting probabilities with model, data shape X: {X.shape}")
        return self.model.predict(X)

def evaluate_model(X, y):
    logging.debug("Evaluating model")
    model = CustomKerasClassifier(input_dim=X.shape[1])
    skf = StratifiedKFold(n_splits=5)
    precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
    logging.debug(f"Precision scores: {precision_scores}, Recall scores: {recall_scores}")
    return precision_scores, recall_scores
