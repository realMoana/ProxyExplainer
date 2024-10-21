import torch
import numpy as np
from tqdm import tqdm

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.EfficiencyEvaluation import EfficiencyEvluation
from ExplanationEvaluation.explainers.ProxyExplainer import PROXYExplainer
from ExplanationEvaluation.explainers.ProxyExplainer_ba2 import PROXYExplainer_ba2
from ExplanationEvaluation.models.model_selector import model_selector



def to_torch_graph(graphs):
    """
    Transforms the numpy graphs to torch tensors depending on the task of the model that we want to explain
    :param graphs: list of single numpy graph
    :return: torch tensor
    """
    return [torch.tensor(g) for g in graphs]


def select_explainer(dataset_name, explainer, model, graphs, features, epochs, lr, reg_coefs, temp=None, sample_bias=None,device='cpu'):

    if explainer == "PROXY":
        if dataset_name == 'ba2motifs':
            return PROXYExplainer_ba2(model, graphs, features, device=device, epochs=epochs, lr=lr, reg_coefs=reg_coefs, temp=temp, sample_bias=sample_bias) 
        else:
            return PROXYExplainer(model, graphs, features, device=device, epochs=epochs, lr=lr, reg_coefs=reg_coefs, temp=temp, sample_bias=sample_bias) 
    
    else:
        raise NotImplementedError("Unknown explainer type")


def run_experiment(inference_eval, auc_eval, explainer, indices):    
    """
    Runs an experiment.
    We generate explanations for given indices and calculate the AUC score.
    :param inference_eval: object for measure the inference speed
    :param auc_eval: a metric object, which calculate the AUC score
    :param explainer: the explainer we wish to obtain predictions from
    :param indices: indices over which to evaluate the auc
    :returns: AUC score, inference speed
    """
    inference_eval.start_prepate()
    
    explainer.prepare(indices)

    inference_eval.start_explaining()
    explanations = []
    for idx in tqdm(indices):
        graph, expl = explainer.explain(idx)
        explanations.append((graph, expl))
    

    inference_eval.done_explaining()

    auc_score = auc_eval.get_score(explanations)
    time_score = inference_eval.get_score(explanations)

    return auc_score, time_score


def replication(config, extension=False, device='cpu'):
    """
    Perform the replication study.
    First load a pre-trained model.
    Then we train our expainer.
    Followed by obtaining the generated explanations.
    And saving the obtained AUC score in a json file.
    :param config: a dict containing the config file values
    :param extension: bool, wheter to use all indices 
    """
    
    graphs, features, labels, _, _, test_mask = load_dataset(config.dataset)
    explanation_labels, indices = load_dataset_ground_truth(config.dataset)
    features = torch.tensor(features)


    if extension: indices = np.argwhere(test_mask).squeeze()
    labels = torch.tensor(labels)
    graphs = to_torch_graph(graphs)


    model = model_selector(config.model, config.dataset, pretrained=True)
    model = model.to(device)
    
    if config.eval_enabled:
        model.eval()

    explainer = select_explainer(config.dataset,
                                config.explainer,
                                model=model,
                                graphs=graphs,
                                features=features,
                                epochs=config.epochs,
                                lr=config.lr,
                                reg_coefs=[config.reg_size,
                                            config.reg_ent],
                                temp=config.temps,
                                sample_bias=config.sample_bias,
                                device=device)

    
    auc_evaluation = AUCEvaluation(explanation_labels, indices) 
    inference_eval = EfficiencyEvluation()

    auc_scores = []
    times = []

    for _, s in enumerate(config.seeds):
        print(f"Run {s} with seed {s}")
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        np.random.seed(s)

        inference_eval.reset()
        auc_score, time_score = run_experiment(inference_eval, auc_evaluation, explainer, indices) 

        auc_scores.append(auc_score)
        print("score:",auc_score)
        times.append(time_score)
        print("time_elased:",time_score)

    auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    inf_time = np.mean(times)
        
    return (auc, auc_std), inf_time
