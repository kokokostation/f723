from f723.tools.dataset.features import FeatureVector


def split_feature_vector(feature_vector, indices):
    return [
        FeatureVector(
            features=feature_vector.features[index],
            target=feature_vector.target[index],
            pdb_ids=feature_vector.pdb_ids[index]) for index in indices]
