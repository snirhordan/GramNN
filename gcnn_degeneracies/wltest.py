import numpy as np

""" USAGE:

Given two ASE Atom frames

first_adj = first_ase.get_all_distances(mic=True)
second_adj = second_ase.get_all_distances(mic=True)

wl_compare(first_adj, second_adj, lab1= first_ase.numbers, lab2=second_ase.numbers)
returns True if the two structures are undistinguishable based on a distance-decorated WL test

"""

def wl_iter(adj, labels=None, eps=1e-8):
    """ Computes Weisfeiler-Lehman-like labels for a graph with weighted edges.
    Distances are converted to ints to avoid discriminating based on numerical noise."""

    # initializes labels to indistinguishable
    nnodes = len(adj)
    if labels is None:
        labels = np.ones(nnodes, dtype=int)

    # finds new set of labels
    uids = []
    for i in range(nnodes):
        # creates a new set of labels by concatenating label/edge pairs, and then hashing
        edgelab = [ (labels[j], int(adj[i,j]/eps)) for j in range(nnodes) ]
        edgelab.sort()
        # also prepends the UID of the selected node
        uids.append( hash(tuple([labels[i]] + edgelab) ) )

    return uids

def label_reduce(lab1, lab2):
    """ Given hashed labels of two graphs, compresses them to the smallest set of ints.
    This avoid runaway changes in the iteration and still distinguishes between graphs that have
    e.g. the same distances multiplied by a different scaling.
    """

    uids = np.unique(np.concatenate([lab1, lab2]))
    uids = dict(zip(uids, range(len(uids))))

    return [uids[l] for l in lab1], [uids[l] for l in lab2]

MAXITER = 1000
def wl_compare(adj1, adj2, niter=-1, lab1 = None, lab2 = None, eps = 1e-8):
    """ Compares two weighted graphs (given in terms of adjacency matrix) based
    on an extended Weisfeiler-Lehman test. Iterates for <niter> iterations, or until
    convergence (negative niter). """

    if niter<0: niter = MAXITER
    oldlab1, oldlab2 = lab1, lab2
    for iiter in range(niter):
        lab1 = wl_iter(adj1, lab1, eps=eps)
        lab2 = wl_iter(adj2, lab2, eps=eps)
        lab1, lab2 = label_reduce(lab1, lab2)
        if iiter > 1:
            if (((np.array(lab1)-np.array(oldlab1))**2).sum() == 0 and
                ((np.array(lab2)-np.array(oldlab2))**2).sum() == 0):
                break
        oldlab1, oldlab2 = lab1, lab2

    lab1.sort()
    lab2.sort()
    return ((np.array(lab1)-np.array(lab2))**2).sum()==0
