import logging

from collections import namedtuple

logger = logging.getLogger(__name__)


Nucleotide = namedtuple('Nucleotide', 'id base chain_id index')


BasePair = namedtuple('BasePair', 'nt_left nt_right type')
BasePairType = namedtuple('BasePairType', 'saenger lw dssr in_ss')


def normalize_pair(nt_left, nt_right):
    if nt_left.chain_id == nt_right.chain_id and nt_left.index > nt_right.index:
        warning_message = 'For base pair ({}, {}): nt_left.index > nt_right.index'.format(nt_left, nt_right)
        logger.warning(warning_message)

        nt_left, nt_right = nt_right, nt_left

    return nt_left, nt_right


def make_base_pair(nt_left, nt_right, bp_type):
    nt_left, nt_right = normalize_pair(nt_left, nt_right)

    return BasePair(nt_left=nt_left, nt_right=nt_right, type=bp_type)


def make_pair(nt_left, nt_right):
    nt_left, nt_right = normalize_pair(nt_left, nt_right)

    return Pair(nt_left=nt_left, nt_right=nt_right)


NucleotideFeatures = namedtuple('NucleotideFeatures', 'secondary_structure base fragment_length fragment_index')
PairFeatures = namedtuple('PairFeatures', 'neighbours_left neighbours_right relation')
Pair = namedtuple('Pair', 'nt_left nt_right')
PairMeta = namedtuple('PairMeta', 'pdb_id pair type')
PairData = namedtuple('PairData', 'features meta')


Chain = namedtuple('Chain', 'pdb_id id nts ss_bps noncanonical_bps')
