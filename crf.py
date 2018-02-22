import json
import random
from typing import List
import time
import numpy as np
import sys
import tbp
from tbp import Graph, Factor
import matplotlib
from matplotlib import pyplot as plt


LETTERS = 'abcdefghijklmnopqrstuvwxyz'

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


    # print("Read {} CRFs".format(len(gs)))
    # return gs

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

    # print("Read marginals for {} graphs".format(len(all_marginals)))
    # return all_marginals

def adjust_temperature(g: Graph, temp):
    for factor in g.factors:
        factor.table = np.exp(np.log(factor.table) / temp)


def classification_accuracy(marg_1, marg_2):
    """
    :param marg_1: List of lists containing marginals
    :param marg_2: List of lists containing marginals
    :return: Rate of matching assignments where each variable is assigned to its highest probability setting.
    """

    # Results sets must contain the same number of variables
    assert len(marg_1) == len(marg_2)

    marg_1_assignments = [np.argmax(x) for x in marg_1]
    marg_2_assignments = [np.argmax(x) for x in marg_2]
    correct = [x == y for x, y in zip(marg_1_assignments, marg_2_assignments)]
    return sum(correct) / len(correct)

def save_plot(results, name):
    """
    :param results: dict of results {r -> ({k -> (err, infer_time)}, decomp_time)}
    """

    # Draw plot
    plt.title(name)
    plt.xlabel("Sample size K")
    plt.xscale('log')
    plt.ylabel("Average classification accuracy")
    # markers = ['-.co', '-ks', ':g^', '-ks', '-.ks', ':g', '-m', '-.y', '--r^']

    for r in results.keys():
        ks = []
        errs = []
        infer_times = []
        for k in results[r][0].keys():
            ks.append(k)
            errs.append(results[r][0][k][0])
            infer_times.append(results[r][0][k][1])
        plt.plot(ks, errs, label='r = {}'.format(r))

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
    with open("plots/sourcedata_%s.json" % file_id, 'w') as f:
        print(json.dumps(results), file=f)


def run_tests(test_group, rs, ks, temps=None, subset=None):
    """
    :param test_group: Number 1-5 indicating which of Ye Nan's tests to run
    :param rs: List of r values (number of components)
    :param ks: List of k values (TBP sample size)
    :param temps: List of temperature adjustment factors (default is [1], i.e. just use the original graph potentials)
    :param subset: List of indices of graphs to choose, since testing all graphs takes a long time. Default (None)
    tests all graphs.
    :return: Dict of results {temp -> {r -> [{k -> [err, infer_time]}, decomp_time]}}
    """

    if temps is None:
        temps = [1]

    graph_filename = 'hocrf/fold0-{0}/fold0-{0}.pot'.format(test_group)
    marg_filename = 'hocrf/fold0-{0}/fold0-{0}.marg'.format(test_group)

    # Results format will be {temp -> {r -> [{k -> [err, infer_time]}, decomp_time]}}
    # Initially, err, infer_time and decomp_time will be lists - later we take the average
    all_results = {}
    for temp in temps:
        results = {}
        for r in rs:
            results[r] = [{}, []]
            for i, (g, true_marg) in enumerate(zip(
                    read_pot(graph_filename, subset=subset), read_marg(marg_filename, subset=subset))):
                print("Graph {} (#{})".format(i+1, subset[i]))

                # Check number of variables and variable cardinalities match between .pot and .marg
                cardinalities = g.get_cardinality_list()
                assert len(cardinalities) == len(true_marg), "Graph {}: Number of variables in .pot ({}) not equal to .marg ({})".format(i, len(cardinalities), len(true_marg))
                assert all(x == len(LETTERS) for x in cardinalities), "Graph {}: .pot cardinalities are {}, expected all {}".format(i, cardinalities, len(LETTERS))
                assert all(len(x) == len(LETTERS) for x in true_marg), "Graph {}: .marg cardinalities are {}, expected all {}".format(i, [len(x) for x in true_marg], len(LETTERS))

                adjust_temperature(g, temp=temp)

                t0 = time.time()
                dg = g.decompose(r=r)
                decomp_time = time.time() - t0
                results[r][1].append(decomp_time)

                for k in ks:
                    if k not in results[r][0]:
                        results[r][0][k] = [[], []]
                    t0 = time.time()
                    marg_est = dg.tbp_marg(k=k)
                    infer_time = time.time() - t0
                    # err = tbp.l1_error(marg_est, true_marg)
                    err = classification_accuracy(marg_est, true_marg)
                    results[r][0][k][0].append(err)
                    results[r][0][k][1].append(infer_time)
                    print('    temp={}, r={}, k={}: {}'.format(temp, r, k, err))
        all_results[temp] = results

    # Take averages
    for temp in temps:
        for r in rs:
            for k in ks:
                # err
                all_results[temp][r][0][k][0] = np.mean(all_results[temp][r][0][k][0])
                # infer_time
                all_results[temp][r][0][k][1] = np.mean(all_results[temp][r][0][k][1])
            # decomp_time
            all_results[temp][r][1] = np.mean(all_results[temp][r][1])

    return all_results


if __name__ == '__main__':
    test_group = sys.argv[1]
    limit = 20
    temps = [1, 0.5, 0.1]

    # Choose which graphs to test at random (just taking the first `limit` graphs from the start of the files is
    # not fair, because they are different handwriting examples of the same word). Rather than load all graphs
    # into memory and take a subset, we choose which graphs to load in advance to save memory.
    n_graphs = 6250  # Number of graphs in the file
    indices = sorted(random.sample(range(n_graphs), limit))

    res = run_tests(
        test_group,
        rs=[2, 5, 10, 100, 1000],
        ks=[10, 100, 1000, 10000, 100000], #, 1000000],
        temps=temps,
        subset=indices,
    )
    for temp in temps:
        save_plot(res[temp], 'fold0-{}-random{}-t={}'.format(test_group, limit, temp))

