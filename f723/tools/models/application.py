import numpy as np

from collections import namedtuple

from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support

ClassificationResult = namedtuple('ClassificationResult', 'target predicted predicted_proba')


def group_k_fold(make_model, feature_vector):
    group_kfold = GroupKFold(n_splits=5)
    group_kfold.get_n_splits(feature_vector.features, feature_vector.target, feature_vector.pdb_ids)

    predicted_proba = np.zeros_like(feature_vector.target, dtype=np.float32)
    predicted = np.zeros_like(feature_vector.target, dtype=np.float32)
    feature_importances = []

    for train_index, test_index in group_kfold.split(
            feature_vector.features, feature_vector.target, feature_vector.pdb_ids):
        X_train, X_test = feature_vector.features[train_index], feature_vector.features[test_index]
        y_train, y_test = feature_vector.target[train_index], feature_vector.target[test_index]

        model = make_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        predicted[test_index] = y_pred
        predicted_proba[test_index] = model.predict_proba(X_test)[:, 1]
        feature_importances.append(model.feature_importances_)

        print(precision_recall_fscore_support(y_test, y_pred))

    return ClassificationResult(
        target=feature_vector.target,
        predicted=predicted,
        predicted_proba=predicted_proba)
