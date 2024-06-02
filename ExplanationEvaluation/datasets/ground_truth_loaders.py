import os
from numpy.random.mtrand import RandomState
from ExplanationEvaluation.datasets.utils import adj_to_edge_index, load_real_dataset, get_graph_data
import numpy as np

def load_mutag_ground_truth(shuffle=True):
    """Load a the ground truth from the mutagenicity dataset.
    Mutag is a large dataset and can thus take a while to load into memory.
    
    :param shuffle: Wheter the data should be shuffled.
    :returns: np.array, np.array, np.array, np.array
    """
    print("Loading MUTAG dataset, this can take a while")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    original_adjs, original_features, original_labels = load_real_dataset(dir_path + '/Mutagenicity/Mutagenicity_')


    print("Loading MUTAG groundtruth, this can take a while")
    path = dir_path + '/Mutagenicity/Mutagenicity_'
    edge_lists, _, edge_label_lists, _ = get_graph_data(path)

    n_graphs = original_adjs.shape[0]
    indices = np.arange(0, n_graphs)
    if shuffle:
        prng = RandomState(42) # Make sure that the permutation is always the same, even if we set the seed different
        shuffled_indices = prng.permutation(indices)
    else:
        shuffled_indices = indices

    # Create shuffled data
    shuffled_adjs = original_adjs[shuffled_indices]
    shuffled_labels = original_labels[shuffled_indices]
    shuffled_edge_list = [edge_lists[i] for i in shuffled_indices]
    shuffled_edge_label_lists = [edge_label_lists[i] for i in shuffled_indices]

    # Transform to edge index
    shuffled_edge_index = adj_to_edge_index(shuffled_adjs)

    return shuffled_edge_index, shuffled_labels, shuffled_edge_list, shuffled_edge_label_lists


def load_dataset_ground_truth(_dataset, test_indices=None):
    if _dataset == "mutag":
        edge_index, labels, edge_list, edge_labels = load_mutag_ground_truth()
        selected = []
        np_edge_list = []
        for gid in range(0, len(edge_index)):
            ed = edge_list[gid]
            ed_np = np.array(ed).T
            np_edge_list.append(ed_np)
            if np.argmax(labels[gid]) == 0 and np.sum(edge_labels[gid]) > 0:
                selected.append(gid)
        np_edge_labels = [np.array(ed_lab) for ed_lab in edge_labels]
        if test_indices is None:
            return (np_edge_list, np_edge_labels), selected
        else:
            all = range(400, 700, 1)
            filtered = [i for i in all if i in test_indices]
            return (np_edge_list, np_edge_labels), filtered
    else:
        print("Dataset does not exist")
        raise ValueError
