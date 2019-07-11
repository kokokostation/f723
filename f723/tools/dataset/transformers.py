import numpy as np

from itertools import groupby

from f723.tools.dataset.features import FeatureVector


def join_features(nucleotide_feature_vector):
    all_features = np.hstack([nucleotide_feature_vector.categorical_features, nucleotide_feature_vector.smooth_features])

    return FeatureVector(
        features=all_features,
        target=nucleotide_feature_vector.target,
        pdb_ids=nucleotide_feature_vector.pdb_ids)


def normalize_features(feature_vector):
    features = feature_vector.features
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    return FeatureVector(features=features, target=feature_vector.target, pdb_ids=feature_vector.pdb_ids)


def pdb_id_segments(pdb_ids):
    borders = np.cumsum([len(list(g)) for _, g in groupby(pdb_ids)])
    borders = np.insert(borders, 0, [0])

    return list(zip(borders[:-1], borders[1:]))


def pack_neighbours(feature_vector, neighbours_num):
    new_features = []
    new_target = []
    new_pdb_ids = []
    new_index = []

    for (start, stop) in pdb_id_segments(feature_vector.pdb_ids):
        for nt_index in range(start + neighbours_num, stop - neighbours_num):
            new_nt_features = []
            neighbours = sorted([i for delta in range(0, neighbours_num + 1) for i in {nt_index + delta, nt_index - delta}])
            for index in neighbours:
                new_nt_features.append(feature_vector.features[index])

            new_features.append(new_nt_features)
            new_target.append(feature_vector.target[neighbours])
            new_pdb_ids.append(feature_vector.pdb_ids[nt_index])
            new_index.append(nt_index)

    return FeatureVector(
        features=np.array(new_features),
        target=np.array(new_target),
        pdb_ids=np.array(new_pdb_ids)), new_index
