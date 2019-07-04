import os
import re
import json

from itertools import chain
from collections import defaultdict

from f723.tools.dataset.entities import Nucleotide, BasePairType, make_base_pair, Chain
from f723.tools.dssr.nrlist import get_nrlist_chains


def get_dssr_data(pdb_id, json_dir):
    fname = '{}.json'.format(pdb_id)

    with open(os.path.join(json_dir, fname), 'r') as infile:
        return json.load(infile)


def get_pdb_ids(json_dir):
    return [os.path.splitext(fname)[0] for fname in os.listdir(json_dir)]


def iterate_dssr_data(json_dir):
    for pdb_id in get_pdb_ids(json_dir):
        yield pdb_id, get_dssr_data(pdb_id, json_dir)


def get_pair(pair_data, nts):
    bp_type = BasePairType(saenger=pair_data['Saenger'], lw=pair_data['LW'], dssr=pair_data['DSSR'])

    return make_base_pair(nts[pair_data['nt1']], nts[pair_data['nt2']], bp_type)


def get_stems(dssr_data):
    nts = get_nts(dssr_data)

    return [[get_pair(pair_data, nts)
             for pair_data in stem_data['pairs']]
            for stem_data in dssr_data.get('stems', [])]


def filter_intrachain_bps(bps, nts_by_chain):
    return [bp for bp in bps if {bp.nt_left, bp.nt_right}.issubset(nts_by_chain[bp.nt_left.chain_id])]


def get_secondary_structure_bps(dssr_data):
    return list(chain.from_iterable(get_stems(dssr_data)))


def get_bps_by_chain(bps, nts_by_chain):
    bps_by_chain = defaultdict(list)

    for bp in filter_intrachain_bps(bps, nts_by_chain):
        bps_by_chain[bp.nt_left.chain_id].append(bp)

    return bps_by_chain


def get_all_bps(dssr_data):
    nts = get_nts(dssr_data)

    return [get_pair(pair_data, nts) for pair_data in dssr_data.get('pairs', [])]


def get_nts(dssr_data):
    return {raw_nt['nt_id']: Nucleotide(
        id=raw_nt['nt_id'],
        base=raw_nt['nt_code'],
        chain_id=raw_nt['chain_name'],
        index=raw_nt['index_chain']) for raw_nt in dssr_data['nts']}


DBN_KEY_REGEX = re.compile('^m1_chain_(?P<chain_id>.*)$')


def get_nts_by_chain(dssr_data):
    dbns = {}
    for key, value in dssr_data['dbn'].items():
        match = DBN_KEY_REGEX.match(key)
        bseq = value['bseq']
        if match is not None and '&' in bseq:
            dbns[match.group('chain_id')] = bseq.index('&')

    nts_by_chain = defaultdict(list)

    for nt in get_nts(dssr_data).values():
        nts_by_chain[nt.chain_id].append(nt)

    for chain_id, num_nts in dbns.items():
        nts_by_chain[chain_id] = nts_by_chain[chain_id][:num_nts]

    return nts_by_chain


def assemble_chains(nrlist_path, json_dir):
    chains = []

    for pdb_id, chain_ids in get_nrlist_chains(nrlist_path).items():
        dssr_data = get_dssr_data(pdb_id, json_dir)

        nts_by_chain = get_nts_by_chain(dssr_data)
        ss_bps_by_chain = get_bps_by_chain(get_secondary_structure_bps(dssr_data), nts_by_chain)
        bps_by_chain = get_bps_by_chain(get_all_bps(dssr_data), nts_by_chain)

        for chain_id in chain_ids:
            ss_bps = set(ss_bps_by_chain[chain_id])
            noncanonical_bps = set(bps_by_chain[chain_id]) - ss_bps
            nts = nts_by_chain[chain_id]

            chains.append(Chain(pdb_id=pdb_id, id=chain_id, nts=nts, ss_bps=ss_bps,
                                noncanonical_bps=noncanonical_bps))

    return chains
