import os
import pickle
import dill

from tqdm import tqdm
from collections import defaultdict
from itertools import chain

from URS import RSS

from f723.tools.dataset.entities import Nucleotide, BasePairType, make_base_pair, Chain
from f723.tools.nrlist import get_nrlist_chains


def make_sec_struct_models(cif_dir, out_dir, pdb_ids, sec_struct_dir):
    for pdb_id in tqdm(pdb_ids):
        cif_file = os.path.join(cif_dir, '{}.cif1'.format(pdb_id))
        out_file = os.path.join(out_dir, '{}.out1'.format(pdb_id))

        model = RSS.SecStruct(cif_file, out_file)

        with open(os.path.join(sec_struct_dir, '{}.pickle'.format(pdb_id)), 'wb') as outfile:
            pickle.dump(model, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def get_sec_struct_model(sec_struct_dir, pdb_id):
    with open(os.path.join(sec_struct_dir, '{}.pickle'.format(pdb_id)), 'rb') as infile:
        return pickle.load(infile)


def get_nts(urs_model):
    nts = {}

    for dssr_id, (chain_id, type_, index) in urs_model.dssrnucls.items():
        if type_ == 'RES':
            nts[dssr_id] = Nucleotide(
                id=dssr_id,
                base=urs_model.chains[chain_id][type_][index]['NAME'],
                chain_id=chain_id,
                index=index)

    return nts


def get_nts_by_chain(urs_model):
    nts_by_chain = defaultdict(list)

    for nt in get_nts(urs_model).values():
        nts_by_chain[nt.chain_id].append(nt)

    return nts_by_chain


def filter_intrachain_bps(bps, nts_by_chain):
    return [bp for bp in bps if {bp.nt_left, bp.nt_right}.issubset(nts_by_chain[bp.nt_left.chain_id])]


def get_bps_by_chain(bps, nts_by_chain):
    bps_by_chain = defaultdict(list)

    for bp in filter_intrachain_bps(bps, nts_by_chain):
        bps_by_chain[bp.nt_left.chain_id].append(bp)

    return bps_by_chain


def get_pair(pair_data, nts):
    pair_type = pair_data['CLASS']
    bp_type = BasePairType(
        saenger=pair_type[0],
        lw=pair_type[1],
        dssr=pair_type[2],
        in_ss=pair_data['STEM'] is not None)

    return make_base_pair(nts[pair_data['NUCL1'][0]], nts[pair_data['NUCL2'][0]], bp_type)


def get_all_bps(urs_model):
    nts = get_nts(urs_model)

    return [get_pair(pair_data, nts)
            for pair_data in urs_model.bpairs
            if pair_data['NUCL1'][1] == 'RES' and pair_data['NUCL2'][1] == 'RES']


def partition_ss_bps(bps):
    partitioned_bps = [[], []]

    for bp in bps:
        partitioned_bps[bp.type.in_ss].append(bp)

    return {
        'noncanonical_bps': partitioned_bps[False],
        'ss_bps': partitioned_bps[True]
    }


def assemble_chains(nrlist_path, cif_dir, out_dir, sec_struct_dir):
    nrlist_chains = get_nrlist_chains(nrlist_path)
    if not os.path.exists(sec_struct_dir):
        os.makedirs(sec_struct_dir)
        make_sec_struct_models(cif_dir, out_dir, nrlist_chains.keys(), sec_struct_dir)

    chains = []

    for pdb_id, chain_ids in tqdm(nrlist_chains.items()):
        urs_model = get_sec_struct_model(sec_struct_dir, pdb_id)

        nts_by_chain = get_nts_by_chain(urs_model)
        bps_by_chain = get_bps_by_chain(get_all_bps(urs_model), nts_by_chain)

        for chain_id in chain_ids:
            partitioned_bps = partition_ss_bps(bps_by_chain[chain_id])
            nts = nts_by_chain[chain_id]

            chains.append(Chain(pdb_id=pdb_id, id=chain_id, nts=nts, **partitioned_bps))

    return chains


def iterate_urs_models(sec_struct_dir):
    for fname in os.listdir(sec_struct_dir):
        pdb_id, _ = os.path.splitext(fname)

        yield pdb_id, get_sec_struct_model(sec_struct_dir, pdb_id)


def get_batch(index, dataset_dir):
    with open(os.path.join(dataset_dir, 'batch_{}'.format(index)), 'rb') as infile:
        return dill.load(infile)


def get_data(dataset_dir):
    return chain.from_iterable((get_batch(i, dataset_dir) for i in tqdm(range(30))))
