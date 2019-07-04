import pandas as pd

from collections import namedtuple, defaultdict


NRListChain = namedtuple('Chain', 'pdb_id id')


def get_nrlist_chains(nrlist_path):
    nrlist = pd.read_csv(nrlist_path, header=None)
    representatives = []

    for raw_representative in nrlist[1]:
        raw_chains = [raw_chain.split('|') for raw_chain in raw_representative.split('+')]
        representatives.append([NRListChain(pdb_id=raw_chain[0].lower(), id=raw_chain[2])
                                for raw_chain in raw_chains])

    for chains in representatives:
        assert len({chain.pdb_id for chain in chains}) == 1

    nrlist_chains = defaultdict(list)

    for chains in representatives:
        for chain in chains:
            nrlist_chains[chain.pdb_id].append(chain.id)

    return nrlist_chains
