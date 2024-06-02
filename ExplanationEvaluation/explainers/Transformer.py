import torch
import numpy as np

class GraphTransformer:
    def __init__(self, graphs, features):
        self.graphs = graphs
        self.features = features
    
    def renew_graph(self, edge_list, feature):
        max_node = edge_list.max() + 1
        new_feature = feature[:max_node, :]
        return edge_list, max_node, new_feature

    def get_union_node_set(self, nodetypes, nmb_type):
        counts = []
        for nodetype in nodetypes:
            this_counts = np.zeros(nmb_type,dtype=np.int32)
            for node in nodetype:
                this_counts[node]+=1
            counts.append(this_counts)
        union = np.max(counts,0)

        unionNodes = []
        for idx,tp in enumerate(union):
            unionNodes.extend([idx]*tp)
        unionNodes = np.array(unionNodes)
        unionFeature = np.zeros((unionNodes.size, nmb_type))
        unionFeature[np.arange(unionNodes.size), unionNodes] = 1
        
        return union,unionNodes,unionFeature

    def map2unionSpace(self, nodes,nodeindex,edgelist):
        newnode = []
        this_nodeindex = nodeindex.copy()
        for nid,node_type in enumerate(nodes):
            newnode.append(this_nodeindex[node_type])
            this_nodeindex[node_type] +=1
        newedgelist = torch.tensor(newnode)[edgelist.cpu()]  
        return newnode,newedgelist

    def process_graphs(self, indices):
        node_types_list = []
        edge_lists = []

        for idx in indices:
            edge_list, _, feature = self.renew_graph(self.graphs[idx], self.features[idx])
            node_types = torch.argmax(feature, dim=-1)
            node_types_list.append(node_types)
            edge_lists.append(edge_list)

        num_types = self.features[0].shape[1]
        type_nmbs,unionNodes,unionFeature = self.get_union_node_set(node_types_list, num_types)
        
        nodeindex = [0]

        for node in type_nmbs:
            nodeindex.append(node + nodeindex[-1])  

        newedgelists = []
        for sampleid in range(len(indices)):
            newnode,newedgelist = self.map2unionSpace(node_types_list[sampleid],nodeindex,edge_lists[sampleid])
            newedgelists.append(newedgelist)
            
        return newedgelists, unionNodes, unionFeature
