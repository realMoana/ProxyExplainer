import os
import torch
from ExplanationEvaluation.models.GNN_paper import GraphGCN as GNN_GraphGCN

def string_to_model(paper, dataset):
    """
    Retrieve the specific neural network model based on the specified paper and dataset.
    
    Args:
        paper (str): Name of the paper whose model architecture to use.
        dataset (str): Name of the dataset to ensure compatibility with the model input and output.
    
    Returns:
        torch.nn.Module: An instance of the model.
    
    Raises:
        NotImplementedError: If the model for the specified paper or dataset is not implemented.
    """
    if paper == "GNN":
        if dataset == "mutag":
            return GNN_GraphGCN(14, 2)
        elif dataset == "ba2":
            return GNN_GraphGCN(10, 2)
        else:
            raise NotImplementedError(f"Model for dataset {dataset} is not implemented under paper {paper}.")
    else:
        raise NotImplementedError(f"Model for paper {paper} is not implemented.")

def get_pretrained_path(paper, dataset):
    """
    Construct the file path for a pretrained model based on the paper and dataset.
    
    Args:
        paper (str): Name of the paper.
        dataset (str): Name of the dataset.
    
    Returns:
        str: File path to the pretrained model parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, "pretrained", paper, dataset, "best_model")

def model_selector(paper, dataset, pretrained=True, return_checkpoint=False):
    """
    Load a model associated with a given paper and dataset, optionally loading pretrained weights.
    
    Args:
        paper (str): Name of the paper whose model to use.
        dataset (str): Dataset on which the model is trained.
        pretrained (bool): Whether to load a pretrained model.
        return_checkpoint (bool): Whether to return the model's state dictionary along with the model.
    
    Returns:
        torch.nn.Module or (torch.nn.Module, dict): The model, and optionally its state dictionary.
    """
    model = string_to_model(paper, dataset)
    if pretrained:
        path = get_pretrained_path(paper, dataset)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model performance: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}")
        if return_checkpoint:
            return model, checkpoint
    return model