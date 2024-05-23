import logging
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression


def evaluate_models(datasets):
    results = []
    for i, dataset in enumerate(datasets):
        X = dataset.drop(columns=['class'])
        y = dataset['class']

        # Logging di debug
        logging.info(f"Dataset {i + 1} - Shape: {dataset.shape}")
        logging.info(f"Features: {X.columns.tolist()}")
        logging.info(f"Class distribution:\n{y.value_counts()}")

        # Check for NaN values
        logging.info(f"Number of NaNs in dataset {i + 1}:\n{dataset.isna().sum()}")

        # Controlla le etichette di classe
        logging.info(f"Class labels in dataset {i + 1}: {y.unique()}")

        model = LogisticRegression(max_iter=500)
        skf = StratifiedKFold(n_splits=5)
        try:
            precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision', error_score='raise')
            recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall', error_score='raise')
            precision = precision_scores.mean()
            recall = recall_scores.mean()

            # Logging dei risultati di precision e recall
            logging.info(f"Dataset {i + 1} - Precision: {precision}, Recall: {recall}")

            results.append((precision, recall))
        except ValueError as e:
            logging.error(f"Error evaluating model on dataset {i + 1}: {e}")
            results.append((0.0, 0.0))
    return results
