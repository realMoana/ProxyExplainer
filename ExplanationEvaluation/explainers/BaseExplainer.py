from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    def __init__(self, model_to_explain, graphs, features, device='cpu'):
        self.model_to_explain = model_to_explain
        self.graphs = [graph.to(device) for graph in graphs]
        self.features = features.to(device)
        self.device = device

    @abstractmethod
    def prepare(self, args):
        """Prepars the explanation method for explaining.
        Can for example be used to train the method"""
        pass

    @abstractmethod
    def explain(self, index):
        """
        Main method for explaining samples
        :param index: index of node/graph in self.graphs
        :return: explanation for sample
        """
        pass

