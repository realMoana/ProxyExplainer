import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv,global_max_pool, global_mean_pool
from tqdm import tqdm
from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge
from ExplanationEvaluation.explainers.Transformer import GraphTransformer
from scipy.sparse import csr_matrix
    
class GNN_MLP_VariationalAutoEncoder(nn.Module):

    def __init__(self, feature_size, output_size):
        super(GNN_MLP_VariationalAutoEncoder, self).__init__()
        self.conv1 = GCNConv(feature_size, 1024)
        self.conv2 = GCNConv(1024, 512)
        self.conv3 = GCNConv(512, 256)
        hidden_dim = 512
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_size)
        )

        self.fc_mu = nn.Linear(256, hidden_dim)
        self.fc_logvar = nn.Linear(256, hidden_dim)

    def encode(self, inputs):
        x, edge_index, edge_weight = inputs
    
        out1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)
        out1 = F.relu(out1)

        out2 = self.conv2(out1, edge_index, edge_weight=edge_weight)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)
        out2 = F.relu(out2)

        out3 = self.conv3(out2, edge_index, edge_weight=edge_weight)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)
        out3 = F.relu(out3)

        input_lin = out3

        mu = self.fc_mu(input_lin)
        logvar = self.fc_logvar(input_lin)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        scale = 1e-2
        eps = torch.randn_like(std) * scale
        return mu + eps * std
    
    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self,inputs,beta, batch=None):
        
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, beta * logvar)
        
        if batch is None:
            out1, _ = torch.max(z, 0)
            out1 = out1.unsqueeze(0)
            out2 = torch.mean(z, 0).unsqueeze(0)
        else:
            out1 = global_max_pool(z, batch)
            out2 = global_mean_pool(z, batch)

        reduce_z = torch.cat([out1, out2], dim=-1)
        recon_x = self.decode(reduce_z)

        return recon_x, mu, logvar


class PROXYExplainer(BaseExplainer):
    def __init__(self, model_to_explain, graphs, features, device='cpu',epochs=30, lr=0.003, temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features,device)
        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.expl_embedding = self.model_to_explain.embedding_size * 2

    def _create_explainer_input(self, pair, embeds):
        row_embeds = embeds[pair[0]]
        col_embeds = embeds[pair[1]]
        input_expl = torch.cat([row_embeds, col_embeds], dim=1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size(),device=self.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def loss_function(self, recon_x, x, mu, logvar, batch_weight_tensor):
        recon_x = batch_weight_tensor * recon_x
        recon_loss= F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def _loss(self, pred, target, mask, reg_coefs):
        
        scale=0.99
        mask = mask*(2*scale-1.0)+(1.0-scale)
        
        cce_loss = F.cross_entropy(pred, target)
        size_loss = torch.sum(mask) * reg_coefs[0]
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = reg_coefs[1] * torch.mean(mask_ent_reg)
        
        return cce_loss + size_loss + mask_ent_loss
    
    def prepare(self, indices=None):
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        self.train(indices or range(len(self.graphs)))

    def train(self, indices=None):
        
        self.explainer_model.train()
        
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))
        transformer = GraphTransformer(self.graphs, self.features)
        
        edgelists = []
        offset = 0
        batch_features = []
        offsets = []
        node_indicator = [] 
        edge_num = []
        
        self.graphs_new = [self.graphs[i] for i in indices]
        if len(self.graphs_new) > 0:
            self.reg_coefs[0] /= len(self.graphs_new)

        for id, i in enumerate(indices):
            edge_list,nmb_node,feature = transformer.renew_graph(self.graphs[i],self.features[i])
            edge_num.append(edge_list.shape[1])
            node_indicator.append(torch.tensor([id]*nmb_node))
            edgelists.append(edge_list+offset)
            offsets.append(offset)
            batch_features.append(feature)
            offset+=int(nmb_node)

        batch_features_tensor_ori = torch.concat(batch_features,0).to(self.device)
        batch_edge_list_ori = torch.concat(edgelists,-1).to(self.device)
        all_one_edge_weights = torch.ones(batch_edge_list_ori.size(1)).to(self.device)
        
        with torch.no_grad():
            embeds = self.model_to_explain.embedding(batch_features_tensor_ori,batch_edge_list_ori,all_one_edge_weights)

        newedgelists, unionNodes, unionFeature = transformer.process_graphs(indices)
        nmb_node = unionNodes.shape[0]
        feature = torch.tensor(unionFeature,dtype=torch.float32)

        self.vae = GNN_MLP_VariationalAutoEncoder(feature.shape[1], feature.shape[0]*feature.shape[0]).to(self.device)
        self.vae.train()
        optimizer_vae = Adam(self.vae.parameters(), lr=1e-4) 

        labels = []
        edgelists = []
        offset = 0
        batch_features = []
        offsets = []
        node_indicator = []
        weight_tensors = []
        vis_edge_list = []
        
        for i, edge_list in enumerate(newedgelists):
            vis_edge_list.append(edge_list)
            node_indicator.append(torch.tensor([i]*nmb_node))
            edgelists.append(edge_list + offset)
            offsets.append(offset)
            batch_features.append(feature)

            offset += int(nmb_node)

            sparseMatrix = csr_matrix((torch.ones(edge_list.shape[1]), edge_list), 
                            shape = (nmb_node,nmb_node))
            label = sparseMatrix.todense()
            label = torch.tensor(label,dtype=torch.float32)
            label = label.view(-1)
            weight_mask = (label == 1)
            labels.append(label)

            nodes = []
            for i in edge_list:
                for j in i:
                    nodes.append(j)
            nodes = list(set(nodes))

            weight_tensor = torch.ones(weight_mask.size(0))

            weight_mask = torch.zeros((nmb_node, nmb_node))
            for i in nodes:
                for j in nodes:
                    weight_mask[i,j] = 1.0
            weight_tensor *= weight_mask.view(-1)
            weight_tensors.append(weight_tensor)

        
        batch_weight_tensor = torch.stack(weight_tensors).to(self.device)
        batch_label = torch.stack(labels).to(self.device)
        batch_features_tensor = torch.concat(batch_features,0).to(self.device)
        batch_edge_list = torch.concat(edgelists,-1).to(self.device) 
        all_one_edge_weights = torch.ones(batch_edge_list.size(1)).to(self.device)
        node_indicator_tensor = torch.concat(node_indicator,-1).to(self.device)
        
        original_pred  = self.model_to_explain(batch_features_tensor, 
                                batch_edge_list,
                                batch=node_indicator_tensor, 
                                edge_weights=all_one_edge_weights)  


        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            t = temp_schedule(e)
            input_expl = self._create_explainer_input(batch_edge_list_ori, embeds).unsqueeze(0)
            sampling_weights = self.explainer_model(input_expl)
            mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()
            edge_weight_delta = 1.0 - mask
            
            vae_epoch = 10
            for a in range(vae_epoch):
                optimizer_vae.zero_grad()
                mask_used = mask.detach() if a < vae_epoch - 1 else mask
                edge_weight_delta_used = edge_weight_delta.detach() if a < vae_epoch - 1 else edge_weight_delta

                recon_batch_mask, _, _ = self.vae((batch_features_tensor, batch_edge_list, mask_used), beta=0, batch=node_indicator_tensor)
                recon_batch, mu, logvar = self.vae((batch_features_tensor, batch_edge_list, edge_weight_delta_used), beta=1, batch=node_indicator_tensor)
                aug_mask = torch.max(recon_batch_mask, recon_batch)
                
                loss_vae = self.loss_function(aug_mask, batch_label, mu, logvar, batch_weight_tensor)
                loss_vae.backward(retain_graph=(a == vae_epoch - 1))
                
                optimizer_vae.step()


            aug_edge_list = []
            aug_edge_weights = []
            offset_new = 0
            
            for i in range(aug_mask.shape[0]):
                adj_matrix = aug_mask[i].reshape(nmb_node, nmb_node)
                edge_list = torch.nonzero(adj_matrix)
                edge_weights = adj_matrix[edge_list[:, 0], edge_list[:, 1]]

                edge_list = edge_list + offset_new
                aug_edge_list.append(edge_list.T)
                aug_edge_weights.append(edge_weights)
                offset_new += nmb_node
                
            aug_edge_list = torch.concat(aug_edge_list,-1).to(self.device) 
            aug_edge_weights = torch.concat(aug_edge_weights,-1).to(self.device)

            masked_pred = self.model_to_explain(batch_features_tensor,
                                                aug_edge_list, 
                                                batch=node_indicator_tensor,
                                                edge_weights=aug_edge_weights) 
            
            loss = self._loss(masked_pred,torch.argmax(original_pred,-1), mask, self.reg_coefs) 
            loss.backward()
            optimizer.step()
        
 
    def explain(self, index):
        index = int(index)
        
        feats = self.features[index].clone().detach().to(self.device)
        graph = self.graphs[index].clone().detach()
        all_one_edge_weights = torch.ones(graph.size(1)).to(self.device)
        embeds = self.model_to_explain.embedding(feats, graph, all_one_edge_weights).detach()

        input_expl = self._create_explainer_input(graph, embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights