import json
import random
from typing import List
import time
import numpy as np
import sys
import tbp
from collections import defaultdict
from tbp import Graph, Factor, DecomposedGraph, DecomposedFactor
import matplotlib
from matplotlib import pyplot as plt

LETTERS = 'abcdefghijklmnopqrstuvwxyz'

def add_decomposed_factors(*args):
    # TODO when this works, implement as + operator for DecomposedFactor instead

    vars = args[0].vars
    for df in args[1:]:
        assert sorted(df.vars) == sorted(vars), "Cannot add DecomposedFactors with different variable sets"

    weights = np.hstack([df.weights for df in args])
    matrices = [np.hstack([df.matrices[df.vars.index(var)] for df in args]) for var in vars]

    return DecomposedFactor(vars, weights, matrices)


def decompose_sparse_factor(factor: Factor, var_order) -> DecomposedFactor:
    # Return a DecomposedFactor equal to Factor.

    # TODO what variable order should we use to build the tree?

    assert sorted(var_order) == sorted(factor.vars)

    # Base case: 1d vector
    if factor.n_vars == 1:
        return DecomposedFactor(
            vars=factor.vars,
            weights=np.array([1]),
            matrices=[factor.table.reshape((len(factor.table), 1))],
        )


    var = var_order[0]
    var_index = factor.vars.index(var)
    var_cardinality = factor.table.shape[var_index]

    res = None

    for i in range(var_cardinality):
        # Get the ith slice (hyperrectangle) of factor.table along the var axis
        # idx is equivalent to something like [:, :, :, i, :, :, :]
        idx = [slice(None)] * factor.n_vars
        idx[var_index] = i
        cur_slice = factor.table[idx]
        assert cur_slice.ndim == factor.n_vars - 1

        if np.any(cur_slice):
            # This slice contains at least one nonzero entry

            min_value = np.min(cur_slice)
            cur_slice -= min_value

            # Recurse to obtain a decomposition for the slice
            child_vars = [x for x in factor.vars if x != var]
            slice_decomposition = decompose_sparse_factor(Factor(child_vars, cur_slice), var_order[1:])

            # slice_decomposition is for the slice only; we need to add a dimension so the shape matches factor
            var_matrix = np.zeros((var_cardinality, len(slice_decomposition.weights)))
            var_matrix[i, :] = 1
            child_decomposition = DecomposedFactor(
                vars=[var] + slice_decomposition.vars,
                weights=slice_decomposition.weights,
                matrices=[var_matrix] + slice_decomposition.matrices,
            )

            # Add the constant term for this slice - this is a single rank-1 tensor covering the entire slice,
            # with values set to 1 and weight set to min_value.
            const_matrices = [np.ones((x, 1)) for x in factor.table.shape]
            const_matrices[var_index] = np.zeros((var_cardinality, 1))
            const_matrices[var_index][i, 0] = 1
            const_term = DecomposedFactor(
                vars=factor.vars,
                weights=np.array([min_value]),
                matrices=const_matrices,
            )

            # Add these rank-1 tensors to the final result
            if res is None:
                res = add_decomposed_factors(child_decomposition, const_term)
            else:
                res = add_decomposed_factors(res, child_decomposition, const_term)

    return res






def decompose_sparse_factor_ye(factor: Factor) -> DecomposedFactor:
    """
    Recursively construct a tensor decomposition for a sparse factor. Views the factor as a tree of height n_vars.
    Leaf nodes represent single non-default values in the original factor. Each leaf node contributes a single rank-1
    tensor to the final decomposition (pointing to a single table cell). The idea is that the rank-1 tensor for a given
    leaf node is given by taking the outer product of the vectors on the path from the root to that leaf node.
    """
    table = factor.table
    # construct a decomposition for the single-variable case
    if table.ndim == 1: 
        n_terms = 1
        weights = np.ones(n_terms)
        matrices = [table.reshape(table.shape[0], 1)]
    # construct a decomposition for (sparse) higher order factors
    else:
        # np.where returns a tuple of length n_vars, where each element is an np.array of axis coordinates. Calling
        # np.transpose on this returns an array of shape (n_nondefault, n_vars) where each row contains the coordinates
        # of a single non-default entry.
        indices = np.transpose(np.where(table != 1.0))
        weights = []
        tensors = []
        construct_rank1_tensors(tensors, weights, factor, indices, 0)
        matrices = [np.array(m).transpose() for m in np.swapaxes(tensors, 0, 1).tolist()]
        for i in range(factor.n_vars):
            assert matrices[i].shape == (factor.cardinalities[i], len(weights))

    df = DecomposedFactor(factor.vars, weights, matrices)
    # Check that the decomposition didn't change the factor
    assert df.expand() == factor
    return df


def construct_rank1_tensors(tensors, weights, factor, indices, depth):
    """
    Recursive part of decompose_sparse_factor.
    :return: None; tensors and weights are added to the respective lists.
    """
    tensor = []
    for d in range(depth):
        v = np.zeros(factor.cardinalities[d])
        v[indices[0][d]] = 1.
        tensor.append(v)

    # at a leaf node (tip of a non-default branch)
    if depth == factor.n_vars:
        assert len(indices) == 1
        tensors.append(tensor)
        weights.append(factor.table[tuple(indices[0])])
    # at an internal node
    else:
        # first construct tensor for the branch with default value
        unique_values = np.unique(indices[:,depth])
        # print("Variable {}/{}: branching factor {}/{}".format(depth, factor.n_vars, len(unique_values), factor.cardinalities[depth]))
        v = np.ones(factor.cardinalities[depth])
        v[unique_values] = 0.
        tensor.append(v)

        for d in range(depth+1, factor.n_vars):
            v = np.ones(factor.cardinalities[d])
            tensor.append(v)

        tensors.append(tensor)
        weights.append(1.)

        # process each remaining branch separately
        for u in unique_values:
            construct_rank1_tensors(tensors, weights, factor,
                                    indices[np.where(indices[:,depth] == u)], depth+1)


# TODO if this works, move it to tbp.py as g.decompose_as_tree()
def decompose_as_tree(g):
    # TODO f.vars here may not be the optimal ordering for the tree decomposition
    return DecomposedGraph([decompose_sparse_factor(f, f.vars) for f in g.factors])

def read_pot(filename, subset=None) -> List[Graph]:
    """
    Read Ye Nan's .pot format and return the CRFs as a list of Graph instances (each .pot file contains several CRFs
    separated by a blank line). The original files contain ^T characters, these should be replaced with spaces first
    e.g. (on OSX) by:
        $ for i in 1 2 3 4 5; do echo $i; sed -i '' -e 's/^T/ /g' hocrf/fold0-$i/fold0-$i.pot; done
    (for linux, in-place sed syntax is slightly different).
    :param subset: List of indices of graphs to choose, default (None) returns all graphs.
    """
    with open(filename, 'r') as f:
        # Maps <tuple of variable indices> -> Factor, so we can quickly determine whether to add the line to an existing
        # factor or create a new factor of ones (there should only be one factor per set of variable indices per graph).

        # TODO if the factor doesn't appear at all, it is never created. Do we need to create factors of all ones?

        factors = {}
        largest_var = None
        n = 0
        for i, line in enumerate(f):
            if not line.strip():
                # If we see a blank line, this indicates the end of this graph, so yield it and then start building
                # the next graph.
                if factors.values():
                    if not subset or n in subset:
                        yield Graph(factors.values())
                        if n == sorted(subset)[-1]:
                            return
                    n += 1
                    factors = {}
                    largest_var = None
                continue

            try:
                fields = line.split()
                if largest_var is not None:
                    assert int(fields[0]) >= largest_var, "Variables should be listed in increasing order"

                largest_var = int(fields[0])
                values = fields[1:-1]
                log_p = float(fields[-1])

                vars = tuple(range(largest_var - len(values) + 1, largest_var + 1))
                indices = tuple([LETTERS.index(v) for v in values])

                if vars in factors:
                    factors[vars].table[indices] = np.exp(log_p)
                else:
                    table = np.ones((len(LETTERS),) * len(vars))
                    table[indices] = np.exp(log_p)
                    factors[vars] = Factor(vars, table)
            except:
                print("Unexpected line format '{}', assuming end of file".format(line))
                break


def read_marg(filename, subset=None):
    """
    Read Ye Nan's .marg format and return a list of graph marginals, each of which is a list of lists (each .mar file
    contains marginals for multiple graphs, separated by a blank line).
    :param subset: List of indices of graphs to choose, default (None) returns all graphs.
    """
    with open(filename, 'r') as f:

        # Order of a-z values taken by variables is given by the first line of the .marg file, and is not
        # alphabetical - remember the order here and use it to reorder marginals into alphabetical order later
        values = f.readline().split()
        val_order = np.argsort(values)
        assert ''.join([values[i] for i in val_order]) == LETTERS

        marginals = []
        n = 0
        for line in f:
            if not line.strip():
                if marginals:
                    if not subset or n in subset:
                        yield marginals
                        if n == sorted(subset)[-1]:
                            return
                    n += 1
                    marginals = []
                continue

            probabilities = [float(p) for p in line.split()]
            # Reorder according to order given on first line of file
            probabilities = [probabilities[i] for i in val_order]
            assert len(probabilities) == len(values)
            marginals.append(probabilities)


def read_label(filename, subset=None):
    seqs = []
    seq = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                seqs.append(seq)
                seq = []
            else:
                seq.append(line.strip().split()[0])
    return np.array(seqs) if subset == None else np.array(seqs)[subset]


def adjust_temperature(g: Graph, temp):
    for factor in g.factors:
        factor.table = np.exp(np.log(factor.table) / temp)


def classify_marg(marg):
    return [LETTERS[np.argmax(x)] for x in marg]


def n_misclassified(labels_1, labels_2):
    incorrect = [x != y for x, y in zip(labels_1, labels_2)]
    return sum(incorrect)


def save_plot(results, ks, temp, name):
    """
    Plot classification error versus sample size k for a given temperature.
    :param results: Dict {series_label -> results}. Series label is a string such as "r=100", results is an object
    as returned by run_tests.
    """

    # Draw plot
    plt.title(name)
    plt.xlabel("Sample size K")
    plt.xscale('log')
    plt.ylabel("Average classification error")
    # markers = ['-.co', '-ks', ':g^', '-ks', '-.ks', ':g', '-m', '-.y', '--r^']

    for series_label in results.keys():
        errs = [results[series_label][(k, temp)]['class_err'] for k in ks]
        plt.plot(ks, errs, label=series_label)

    # # Add padding so we can see the markers
    # xticks, xticklabels = plt.xticks()
    # xmin = (3 * xticks[0] - xticks[1]) / 2.
    # xmax = (3 * xticks[-1] - xticks[-2]) / 2.
    # plt.xlim(xmin, xmax)
    # plt.xticks(xticks)

    plt.legend(loc='best', ncol=3)
    file_id = time.time()
    plot_filename = "plots/%s-%s.png" % (name, file_id)
    plt.savefig(plot_filename)
    plt.gcf().clear()
    print("%s saved\n" % plot_filename)
    # TODO currently can't json.dumps, probably because of the (k, temp) tuple key
    # with open("plots/sourcedata_%s.json" % file_id, 'w') as f:
    #     print(json.dumps(results), file=f)


def make_ils_decompose_fn(r):
    def decompose_fn(g):
        return g.decompose(r=r)
    return decompose_fn


def make_tree_decompose_fn():
    def decompose_fn(g):
        return decompose_as_tree(g)
    return decompose_fn


def run_tests(test_group, decompose_fn, ks, temps, subset=None):
    """
    :param test_group: Number 1-5 indicating which of Ye Nan's tests to run
    :param decompose_fn: TODO
    :param ks: List of k values (TBP sample size)
    :param temps: List of temperature adjustment factors (default is [1], i.e. just use the original graph potentials)
    :param subset: List of indices of graphs to choose, since testing all graphs takes a long time. Default (None)
    tests all graphs.
    :return: Dict of results, each of which is an dict {params -> results}. results is a dict with keys (decomp_time,
    infer_time, marg_err, class_err). params is a tuple (k, temp).
    """

    graph_filename = 'hocrf/fold0-{0}/fold0-{0}.pot'.format(test_group)
    marg_filename = 'hocrf/fold0-{0}/fold0-{0}.marg'.format(test_group)
    label_filename = 'hocrf/labels/fold0-{0}.ts'.format(test_group)

    # Initially, all results will be lists (one entry for each graph) - later we take the average

    results = {}
    n_vars_total = 0
    for temp in temps:
        for i, (g, true_marg, true_labels) in enumerate(zip(
                read_pot(graph_filename, subset=subset),
                read_marg(marg_filename, subset=subset),
                read_label(label_filename, subset=subset),
        )):
            print("Graph {} (#{})".format(i+1, subset[i]))

            # Check number of variables and variable cardinalities match between .pot and .marg
            cardinalities = g.get_cardinality_list()
            assert len(cardinalities) == len(true_marg), "Graph {}: Number of variables in .pot ({}) not equal to .marg ({})".format(i, len(cardinalities), len(true_marg))
            assert all(x == len(LETTERS) for x in cardinalities), "Graph {}: .pot cardinalities are {}, expected all {}".format(i, cardinalities, len(LETTERS))
            assert all(len(x) == len(LETTERS) for x in true_marg), "Graph {}: .marg cardinalities are {}, expected all {}".format(i, [len(x) for x in true_marg], len(LETTERS))

            adjust_temperature(g, temp=temp)
            t0 = time.time()
            dg = decompose_fn(g)
            decomp_time = time.time() - t0
            n_vars_total += len(true_labels)

            for k in ks:
                if (k, temp) not in results:
                    results[(k, temp)] = defaultdict(list)
                t0 = time.time()
                marg_est = dg.tbp_marg(k=k)
                infer_time = time.time() - t0
                marg_err = tbp.l1_error(marg_est, true_marg)
                est_labels = classify_marg(marg_est)
                class_err = n_misclassified(est_labels, true_labels)

                results[(k, temp)]['infer_time'].append(infer_time)
                results[(k, temp)]['decomp_time'].append(decomp_time)
                results[(k, temp)]['marg_err'].append(marg_err)
                results[(k, temp)]['class_err'].append(class_err)
                # print('    actual: {}, predicted: {}'.format(true_labels, est_labels))
                print('    temp={}, k={}:\tactual={}, predicted={}, decomp_time={}, infer_time={}, marg_err={}, class_err={}'
                      .format(temp, k, ''.join(true_labels), ''.join(est_labels), decomp_time, infer_time, marg_err, class_err))

    # Take averages
    for temp in temps:
        for k in ks:
            for results_key in ('infer_time', 'decomp_time', 'marg_err'):
                results[(k, temp)][results_key] = np.mean(results[(k, temp)][results_key])
            # For classification error, we average over all graphs rather than averaging over each graph individually
            # first (i.e. long words will contribute more error than short words, which is not the case for marg_err)
            for results_key in ('class_err',):
                results[(k, temp)][results_key] = np.sum(results[(k, temp)][results_key]) / n_vars_total

    return results

# def run_tests(test_group, rs, ks, temps=None, subset=None):
#     """
#     :param test_group: Number 1-5 indicating which of Ye Nan's tests to run
#     :param rs: List of r values (number of components)
#     :param ks: List of k values (TBP sample size)
#     :param temps: List of temperature adjustment factors (default is [1], i.e. just use the original graph potentials)
#     :param subset: List of indices of graphs to choose, since testing all graphs takes a long time. Default (None)
#     tests all graphs.
#     :return: Dict of results {temp -> {r -> [{k -> [err, infer_time]}, decomp_time]}}
#     """
#
#     if temps is None:
#         temps = [1]
#
#     graph_filename = 'hocrf/fold0-{0}/fold0-{0}.pot'.format(test_group)
#     marg_filename = 'hocrf/fold0-{0}/fold0-{0}.marg'.format(test_group)
#     label_filename = 'hocrf/labels/fold0-{0}.ts'.format(test_group)
#
#     # Results format will be {temp -> {r -> [{k -> [err, infer_time]}, decomp_time]}}
#     # Initially, err, infer_time and decomp_time will be lists - later we take the average
#     all_results = {}
#     use_tree_decomposition = True
#     for temp in temps:
#         results = {}
#         for r in rs:
#             results[r] = [{}, []]
#             tot_len = 0
#             for i, (g, true_marg, seq) in enumerate(zip(
#                     read_pot(graph_filename, temp=temp, decompose=use_tree_decomposition, subset=subset),
#                     read_marg(marg_filename, subset=subset),
#                     read_label(label_filename, subset=subset),
#                     )):
#                 print("Graph {} (#{})".format(i+1, subset[i]))
#
#                 # Check number of variables and variable cardinalities match between .pot and .marg
#                 cardinalities = g.get_cardinality_list()
#                 assert len(cardinalities) == len(true_marg), "Graph {}: Number of variables in .pot ({}) not equal to .marg ({})".format(i, len(cardinalities), len(true_marg))
#                 assert all(x == len(LETTERS) for x in cardinalities), "Graph {}: .pot cardinalities are {}, expected all {}".format(i, cardinalities, len(LETTERS))
#                 assert all(len(x) == len(LETTERS) for x in true_marg), "Graph {}: .marg cardinalities are {}, expected all {}".format(i, [len(x) for x in true_marg], len(LETTERS))
#
#
#                 if use_tree_decomposition:
#                     dg = g
#                     decomp_time = 0. # very fast...
#                 else:
#                     adjust_temperature(g, temp=temp)
#                     t0 = time.time()
#                     dg = g.decompose(r=r)
#                     decomp_time = time.time() - t0
#                 results[r][1].append(decomp_time)
#                 tot_len += len(seq)
#
#                 for k in ks:
#                     if k not in results[r][0]:
#                         results[r][0][k] = [[], []]
#                     t0 = time.time()
#                     marg_est = dg.tbp_marg(k=k)
#                     infer_time = time.time() - t0
#                     # err = tbp.l1_error(marg_est, true_marg)
#                     # err = classification_accuracy(marg_est, true_marg)
#                     pred = np.array(list(LETTERS))[np.array(marg_est).argmax(axis=1)]
#                     err = (pred != seq).sum()
#                     results[r][0][k][0].append(err)
#                     results[r][0][k][1].append(infer_time)
#                     if use_tree_decomposition:
#                         print('    temp={}, k={}, err={}, infer_time={}'
#                             .format(temp, k, err, infer_time))
#                     else:
#                         print('    temp={}, r={}, k={}, err={}, decomp_time={}, infer_time={}'
#                             .format(temp, r, k, err, decomp_time, infer_time))
#         all_results[temp] = results
#
#     # Take averages
#     for temp in temps:
#         for r in rs:
#             for k in ks:
#                 # err
#                 all_results[temp][r][0][k][0] = np.sum(all_results[temp][r][0][k][0]) / tot_len
#                 # infer_time
#                 all_results[temp][r][0][k][1] = np.mean(all_results[temp][r][0][k][1])
#             # decomp_time
#             all_results[temp][r][1] = np.mean(all_results[temp][r][1])
#
#     print(all_results)
#     return all_results
#

def test():
    x = Factor([0,1,2], np.array([
        [
            [1,2,3],
            [4,5,6]
        ],
        [
            [-1,0,3],
            [4,44,6]
        ],
    ]))
    res = decompose_sparse_factor(x, [2,0,1])
    return res


def main():
    test_group = sys.argv[1]
    limit = 20
    ks = [100, 1000, 10000, 100000]

    # Choose which graphs to test at random (just taking the first `limit` graphs from the start of the files is
    # not fair, because they are different handwriting examples of the same word). Rather than load all graphs
    # into memory and take a subset, we choose which graphs to load in advance to save memory.
    n_graphs = 6250  # Number of graphs in the file
    indices = sorted(random.sample(range(n_graphs), limit))

    res_tree = run_tests(
        test_group,
        decompose_fn=make_tree_decompose_fn(),
        ks=ks,
        temps=[1.],
        subset=indices,
    )
    res_ils = run_tests(
        test_group,
        decompose_fn=make_ils_decompose_fn(r=100),
        ks=ks,
        temps=[1.],
        subset=indices,
    )
    save_plot({'ILS (r=100)': res_ils, 'TREE': res_tree}, ks, 1., 'fold0-{}-random{}-tree'.format(test_group, limit))


if __name__ == '__main__':
    main()


