import os
import pickle
import numpy as np
from numpy.random import RandomState
from ExplanationEvaluation.datasets.utils import adj_to_edge_index, load_real_dataset


def load_graph_dataset(dataset_name, shuffle=True):
    """Load and optionally shuffle a graph dataset.
    
    Parameters:
    - dataset_name (str): Name of the dataset to load.
    - shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
    Tuple containing edge indices, features, labels, and masks for training, validation, and testing.
    
    Raises:
    FileNotFoundError: If the dataset pickle file does not exist and cannot be created.
    NotImplementedError: If the dataset is unknown.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dir_path, 'pkls', f"{dataset_name}.pkl")

    if not os.path.exists(dataset_path):
        adjs, features, labels = load_real_dataset(os.path.join(dir_path + '/Mutagenicity/Mutagenicity_'))
    else:
        with open(dataset_path, 'rb') as fin:
            adjs, features, labels = pickle.load(fin)

    if dataset_name.lower() != "mutag": 
        # raise NotImplementedError("Unknown dataset")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = dir_path + '/pkls/' + "BA-2motif" + '.pkl'
        with open(path, 'rb') as fin:
            adjs, features, labels = pickle.load(fin)

    n_graphs = adjs.shape[0]
    indices = np.arange(n_graphs)
    if shuffle:
        prng = RandomState(42) 
        indices = prng.permutation(indices)


    adjs = adjs[indices]
    features = features[indices].astype('float32')
    labels = labels[indices]

    n_train = int(n_graphs * 0.8)
    n_val = int(n_graphs * 0.9)
    train_mask = np.zeros(n_graphs, dtype=bool)
    val_mask = np.zeros(n_graphs, dtype=bool)
    test_mask = np.zeros(n_graphs, dtype=bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_val] = True
    test_mask[n_val:] = True

    edge_index = adj_to_edge_index(adjs)

    return edge_index, features, labels, train_mask, val_mask, test_mask


def load_dataset(dataset, skip_preprocessing=False, shuffle=True):
    """High-level function to load the dataset, optionally skipping preprocessing.
    
    Parameters:
    - dataset (str): Name of the dataset to load.
    - skip_preprocessing (bool): Whether to skip converting the adjacency matrix to edge indices.
    - shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
    Multiple numpy arrays (edge indices, features, labels, and masks).
    """
    print(f"Loading {dataset} dataset")
    data = load_graph_dataset(dataset, shuffle)
    if skip_preprocessing:
        return data[:-1] 
    return data