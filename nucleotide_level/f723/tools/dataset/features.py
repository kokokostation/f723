import numpy as np
import json
import pickle

from tqdm import tqdm
from collections import defaultdict, namedtuple
from scipy.sparse import lil_matrix

from f723.tools.dataset.entities import NucleotideFeatures
from f723.tools.urs.extraction import get_sec_struct_model, assemble_chains


NucleotideFeatureVector = namedtuple(
    'NucleotideFeatureVector',
    'categorical_features smooth_features target pdb_ids target_matrix relation_matrix stem_matrix')
FeatureVector = namedtuple('FeatureVector', 'features target pdb_ids')


class NucleotideFeaturesExtractor:
    SECONDARY_STRUCTURES = ['BC', 'BI', 'BP', 'HC', 'HI', 'HP', 'IC', 'II', 'IP', 'JC', 'JI', 'JP', 'S']
    BASES = ['a', 'u', 'g', 'c']
    RELATIONS = ['LC', 'LR', 'SM']
    CATEGORICAL_FEATURES_LEN = len(SECONDARY_STRUCTURES) + len(BASES) + 1
    SMOOTH_FEATURES_LEN = 2

    def __init__(self, nrlist_path, cif_dir, out_dir, sec_struct_dir, chains_for_classification_path, max_pair_dist):
        self.nrlist_path = nrlist_path
        self.cif_dir = cif_dir
        self.out_dir = out_dir
        self.sec_struct_dir = sec_struct_dir
        self.chains_for_classification_path = chains_for_classification_path
        self.max_pair_dist = max_pair_dist

    def make_chains_by_pdb_id(self):
        with open(self.chains_for_classification_path, 'r') as infile:
            chains_for_classification = json.load(infile)

        all_chains = {(chain.pdb_id, chain.id): chain
                      for chain in assemble_chains(self.nrlist_path, self.cif_dir, self.out_dir, self.sec_struct_dir)}

        chains_by_pdb_id = defaultdict(list)
        for pdb_id, chain_id in chains_for_classification:
            chains_by_pdb_id[pdb_id].append(all_chains[pdb_id, chain_id])

        return chains_by_pdb_id

    def make_nucleotide_features(self, urs_model, chain):
        nucleotide_features = []
        nucleotide_index = {}

        for index, nt in enumerate(chain.nts):
            chain_entry = urs_model.chains[chain.id]['RES'][nt.index]

            secondary_structure = urs_model.NuclSS(nt.id)
            base = nt.base.lower()

            if chain_entry['WING']:
                wing = urs_model.wings['LU'][chain_entry['WING'] - 1]
                fragment_length = wing['LEN']
                fragment_index = nt.index - wing['START'][2]
            elif chain_entry['THREAD']:
                thread = urs_model.threads[chain_entry['THREAD'] - 1]
                fragment_length = thread['LEN']
                fragment_index = nt.index - thread['START'][2]
            else:
                raise ValueError

            nucleotide_features.append(NucleotideFeatures(
                secondary_structure=secondary_structure,
                base=base,
                fragment_length=fragment_length,
                fragment_index=fragment_index))
            nucleotide_index[nt] = index

        return nucleotide_features, nucleotide_index

    def make_nucleotide_feature_vectors(self, nucleotide_features):
        categorical_features = []
        smooth_features = []

        for nf in nucleotide_features:
            categorical_row = []
            smooth_row = [nf.fragment_length, nf.fragment_index]

            nucleotide_secondary_structures = nf.secondary_structure.split(';')
            for secondary_structure in self.SECONDARY_STRUCTURES:
                categorical_row.append(int(secondary_structure in nucleotide_secondary_structures))

            for base in self.BASES:
                categorical_row.append(int(nf.base == base))

            categorical_row.append(int(nf.base in self.BASES))

            categorical_features.append(categorical_row)
            smooth_features.append(smooth_row)

        return categorical_features, smooth_features

    def make_dataset(self):
        chains_by_pdb_id = self.make_chains_by_pdb_id()

        all_chains_length = sum([len(chain.nts) for chains in chains_by_pdb_id.values() for chain in chains])

        all_categorical_features = np.zeros(dtype=np.uint8, shape=(all_chains_length, self.CATEGORICAL_FEATURES_LEN))
        all_smooth_features = np.zeros(dtype=np.uint8, shape=(all_chains_length, self.SMOOTH_FEATURES_LEN))
        all_pdb_ids = np.zeros(dtype=np.object, shape=(all_chains_length,))
        all_target = np.zeros(dtype=np.bool, shape=(all_chains_length,))
        target_matrix = lil_matrix((all_chains_length, all_chains_length), dtype=np.bool)
        relation_matrix = lil_matrix((all_chains_length, all_chains_length), dtype=np.uint8)
        stem_matrix = lil_matrix((all_chains_length, all_chains_length), dtype=np.bool)

        with open('/home/mikhail/bioinformatics/data/dataset_all_60/realtions_data', 'rb') as infile:
            relation_data = pickle.load(infile)

        current_index = 0

        for pdb_id, chains in tqdm(chains_by_pdb_id.items()):
            urs_model = get_sec_struct_model(self.sec_struct_dir, pdb_id)

            for chain in chains:
                chain_len = len(chain.nts)
                current_slice = slice(current_index, current_index + chain_len)

                nucleotide_features, nucleotide_index = self.make_nucleotide_features(urs_model, chain)
                categorical_features, smooth_features = self.make_nucleotide_feature_vectors(nucleotide_features)

                all_categorical_features[current_slice, :] = categorical_features
                all_smooth_features[current_slice, :] = smooth_features

                for bp in chain.noncanonical_bps:
                    left_index = current_index + nucleotide_index[bp.nt_left]
                    right_index = current_index + nucleotide_index[bp.nt_right]

                    all_target[left_index] = all_target[right_index] = True
                    target_matrix[left_index, right_index] = target_matrix[right_index, left_index] = True

                for bp in chain.ss_bps:
                    left_index = current_index + nucleotide_index[bp.nt_left]
                    right_index = current_index + nucleotide_index[bp.nt_right]

                    stem_matrix[left_index, right_index] = stem_matrix[right_index, left_index] = True

                for left_index, right_index, relation in relation_data[(pdb_id, chain.id)]:
                    li = current_index + left_index
                    ri = current_index + right_index

                    relation = self.RELATIONS.index(relation)
                    relation_matrix[li, ri] = relation
                    relation_matrix[ri, li] = relation

                all_pdb_ids[current_slice] = '{}.{}'.format(pdb_id, chain.id)

                current_index += chain_len

        return NucleotideFeatureVector(
            categorical_features=all_categorical_features,
            smooth_features=all_smooth_features,
            pdb_ids=all_pdb_ids,
            target=all_target,
            target_matrix=target_matrix,
            stem_matrix=stem_matrix,
            relation_matrix=relation_matrix)
