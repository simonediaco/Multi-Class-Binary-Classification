from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np

class CustomKerasMultiClassClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, num_classes, epochs=150, batch_size=10):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model = self.build_model()
        y_encoded = to_categorical(y, num_classes=self.num_classes)
        self.model.fit(X, y_encoded, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

    def predict_proba(self, X):
        return self.model.predict(X)

def evaluate_multi_class_model(X, y, num_classes):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    model = CustomKerasMultiClassClassifier(input_dim=X.shape[1], num_classes=num_classes)
    skf = StratifiedKFold(n_splits=5)
    accuracy_scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy')
    precision_scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='precision_weighted')
    recall_scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='recall_weighted')
    return accuracy_scores, precision_scores, recall_scores
