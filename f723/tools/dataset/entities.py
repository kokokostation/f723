import logging

from collections import namedtuple
from enum import Enum

logger = logging.getLogger(__name__)


Nucleotide = namedtuple('Nucleotide', 'id base chain_id index')


BasePair = namedtuple('BasePair', 'nt_left nt_right type')
BasePairType = namedtuple('BasePairType', 'saenger lw dssr')


def make_base_pair(nt_left, nt_right, bp_type):
    if nt_left.chain_id == nt_right.chain_id and nt_left.index > nt_right.index:
        warning_message = 'For base pair ({}, {}): nt_left.index > nt_right.index'.format(nt_left, nt_right)
        logger.warning(warning_message)

        nt_left, nt_right = nt_right, nt_left

    return BasePair(nt_left=nt_left, nt_right=nt_right, type=bp_type)


class PairType(Enum):
    SECONDARY_STRUCTURE = 'secondary_structure'
    NON_CANONICAL = 'non_canonical'
    NO_BOND = 'no_bond'


NucleotideFeatures = namedtuple('NucleotideFeatures', 'secondary_structure base fragment_length fragment_index')
PairFeatures = namedtuple('Features', '')
PairMeta = namedtuple('PairMeta', 'pdb_id bp type')
Pair = namedtuple('Pair', 'features meta')


Chain = namedtuple('Chain', 'pdb_id id nts ss_bps noncanonical_bps')
