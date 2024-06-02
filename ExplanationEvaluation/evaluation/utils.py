import numpy as np
from sklearn.metrics import roc_auc_score

def evaluation_auc(explanations, explanation_labels, indices):
    """
    Evaluate the AUC score for given explanations and corresponding ground truth labels.
    
    Args:
        explanations (list): List of tuples (graph, mask) for each explanation where:
                             - graph is the adjacency matrix of the graph,
                             - mask is the prediction mask for edges.
        explanation_labels (tuple of np.arrays): A tuple containing:
                             - List of ground truth edge lists for each graph,
                             - Corresponding edge labels for each graph.
        indices (list): Indices indicating which explanations and labels to evaluate.

    Returns:
        float: The computed area under the ROC curve score.

    Notes:
        This function excludes self-loops in the evaluation and assumes the masks and ground truths
        are aligned and correspond to the indices provided.
    """
    predictions = []
    ground_truth = []

    for idx, n in enumerate(indices): 
        mask = explanations[idx][1].detach().cpu().numpy()
        graph = explanations[idx][0].detach().cpu().numpy()

        edge_list = explanation_labels[0][n]
        edge_labels = explanation_labels[1][n]

        for edge_idx in range(0, edge_labels.shape[0]): 
            edge_ = edge_list.T[edge_idx]
            if edge_[0] == edge_[1]:  
                continue
            t = np.where((graph.T == edge_.T).all(axis=1))

            predictions.append(mask[t][0])
            ground_truth.append(edge_labels[edge_idx])

    return roc_auc_score(ground_truth, predictions) if ground_truth and predictions else None
