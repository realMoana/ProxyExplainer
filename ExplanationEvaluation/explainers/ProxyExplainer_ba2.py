import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch_geometric.nn import Sequential, GCNConv,global_max_pool, global_mean_pool
from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge
from scipy.sparse import csr_matrix

class VGAE(nn.Module):
    def __init__(self, feature_size, output_size):
        super(VGAE, self).__init__()
        self.conv1 = GCNConv(feature_size, 1024) 
        self.conv2 = GCNConv(1024, 512)
        self.conv3 = GCNConv(512, 256) 
        hid_dim = 1024 
        
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim*2,hid_dim*2),
            nn.ReLU(),
            nn.Linear(hid_dim*2,output_size))

        self.fc_mu = nn.Linear(256, hid_dim) 
        self.fc_logvar = nn.Linear(256, hid_dim) 
    
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
        std = logvar.mul(0.5).exp_()
        scale = 1e-2 
        eps = torch.randn_like(std) * scale 
        return eps.mul(std).add_(mu)

    def decode(self, inputs):
        return F.sigmoid(self.decoder(inputs))
    
    def forward(self,inputs,beta, batch=None):

        mu, logvar = self.encode(inputs)
        embed = self.reparameterize(mu, beta * logvar)

        if batch is None:
            out1,_ = torch.max(embed,0)
            out1 = torch.unsqueeze(out1,0)
            out2 = torch.unsqueeze(torch.mean(embed,0),0)
        else:
            out1 = global_max_pool(embed, batch)
            out2 = global_mean_pool(embed, batch)

        reduce_z = torch.cat([out1, out2], dim=-1)
        recon_x = self.decode(reduce_z) 

        return recon_x, mu, logvar


class PROXYExplainer_ba2(BaseExplainer):
    
    def __init__(self, model_to_explain, graphs, features, device='cpu',epochs=30, lr=0.003, temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features, device)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias

        self.expl_embedding = self.model_to_explain.embedding_size * 2
        self.device = device

    def _create_explainer_input(self, pair, embeds):
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        if training:
            bias = bias + 0.0001  
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size(),device=self.device) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1] 

        scale = 0.99
        mask = mask*(2*scale-1.0)+(1.0-scale)

        size_loss = torch.sum(mask) * size_reg 
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg) 

        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self,indices=None):
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

        if indices is None: 
            indices = range(0, self.graphs.size(0))

        self.train(indices=indices)
    
    def aug_with_vae(self, graph,sparse_block_tensor, graph_size=0):
        feats,edge_list,edge_weights = graph

        sparse_tensor = torch.sparse.FloatTensor(edge_list, edge_weights, torch.Size([feats.shape[0], feats.shape[0]]))

        input_tensor = sparse_block_tensor+sparse_tensor
        input_tensor_coalesce = input_tensor.coalesce()
        values = input_tensor_coalesce.values()-1e-6


        edge_weight_delta = 1.0-edge_weights
        sparse_tensor_delta = torch.sparse.FloatTensor(edge_list, edge_weight_delta, torch.Size([feats.shape[0], feats.shape[0]]))
        input_tensor_delta = sparse_block_tensor + sparse_tensor_delta
        input_tensor_coalesce_delta = input_tensor_delta.coalesce()
        values_delta = input_tensor_coalesce_delta.values() - 1e-6
        batch_values_delta = values_delta.view(-1, graph_size * graph_size)
       
        return edge_weight_delta, values.view(-1, graph_size * graph_size), batch_values_delta, edge_weights
    
    def loss_function(self, recon_x, x, mu, logvar, batch_weight_tensor_ori, batch_weight_tensor, batch_norm, aug_edge_weights_mask, values_mask):

        recon_loss= F.binary_cross_entropy(recon_x, x, reduction='mean')
        kl_loss = (-0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1)).mean()
        aug_edge_weights_mask = batch_weight_tensor * aug_edge_weights_mask
        mse_loss = F.mse_loss(aug_edge_weights_mask, values_mask, reduction = "mean") * torch.mean(batch_norm) 
        alpha = 0.1
        
        return recon_loss + alpha * kl_loss + mse_loss

        
    def train(self, indices=None):

        self.explainer_model.train()

        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))

        labels = []
        edgelists = []
        offset = 0
        full_rows = []
        full_cols = []
        batch_features = []
        offsets = []
        node_indicator = []
        edge_num = []
        
        nmb_node = self.features[0].shape[0]
        self.graphs_train = self.graphs
        self.reg_coefs[0]/= len(indices)
        
        weight_tensors = []
        norms = []
        weight_tensors_ori = []
        
        for id, i in enumerate(indices):
            
            feature = self.features[i] 
            edge_list = self.graphs_train[i] 
            
            
            sparseMatrix = csr_matrix((torch.ones(edge_list.shape[1]), edge_list.cpu()), 
                            shape = (nmb_node,nmb_node))
            label = sparseMatrix.todense()
            label = torch.tensor(label,dtype=torch.float32)
            label = label.view(-1)
            weight_mask = (label == 1)
            labels.append(label)
            
            
            norm = nmb_node  * nmb_node  / float((nmb_node  * nmb_node  - sparseMatrix.sum()) * 2)
            pos_weight = float(nmb_node * nmb_node - sparseMatrix.sum()) / sparseMatrix.sum()
            

            weight_tensor = torch.ones(weight_mask.size(0))
            weight_tensor[weight_mask] = pos_weight
            
            weight_tensor_ori = torch.ones(weight_mask.size(0)) * 0.1
            weight_tensor_ori[weight_mask] = 1
            weight_tensors_ori.append(weight_tensor_ori)
            
            
            weight_tensors.append(weight_tensor)
            norms.append(norm)
            

            node_indicator.append(torch.tensor([id]*nmb_node))

            edge_num.append(edge_list.shape[1])
            
            edgelists.append(edge_list+offset)
            offsets.append(offset)
            batch_features.append(feature)

            _temp_arange_nmb_node = torch.arange(nmb_node)
            _temp_matrix_nmb_node = _temp_arange_nmb_node.repeat(nmb_node,1)
            _full_col = _temp_matrix_nmb_node.view(-1)+offset
            _full_row = _temp_matrix_nmb_node.transpose(1,0).reshape(-1)+offset
            full_rows.append(_full_row)
            full_cols.append(_full_col)

            offset+=int(nmb_node)
            
        batch_weight_tensor = torch.stack(weight_tensors).to(self.device)
        batch_norm = torch.tensor(norms).to(self.device)
        
        batch_weight_tensor_ori = torch.stack(weight_tensors_ori).to(self.device)
        
        batch_features_tensor = torch.concat(batch_features,0).to(self.device)
        batch_edge_list = torch.concat(edgelists,-1).to(self.device)
        
        batch_full_rows = torch.concat(full_rows)
        batch_full_cols = torch.concat(full_cols)
        batch_fully_graphs = torch.stack([batch_full_rows,batch_full_cols]).to(self.device)
        
        all_one_edge_weights = torch.ones(batch_edge_list.size(1)).to(self.device)
        node_indicator_tensor = torch.concat(node_indicator,-1).to(self.device)
        
        batch_label = torch.stack(labels).to(self.device)


        sparse_block_tensor = torch.sparse.FloatTensor(batch_fully_graphs,
                                                    torch.ones(batch_fully_graphs.shape[1],device=self.device)*1e-6, 
                                                    torch.Size([batch_features_tensor.shape[0], batch_features_tensor.shape[0]])).to(self.device)

        
        with torch.no_grad():
            embeds = self.model_to_explain.embedding(batch_features_tensor,batch_edge_list,all_one_edge_weights)
            original_pred, _ = self.model_to_explain(batch_features_tensor, 
                                            batch_edge_list,
                                            batch=node_indicator_tensor, 
                                            edge_weights=all_one_edge_weights)
            
        
        self.vae = VGAE(10, 625).to(self.device)
        self.vae.train()
        optimizer_vae = Adam(self.vae.parameters(), lr=1e-3)

        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()

            t = temp_schedule(e)
            input_expl = self._create_explainer_input(batch_edge_list, embeds).unsqueeze(0)
            sampling_weights = self.explainer_model(input_expl)
            mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

            batch_graph_delta, values_mask, batch_graph_delta_new,values_mask_new  = self.aug_with_vae((batch_features_tensor,batch_edge_list,mask),sparse_block_tensor, graph_size = nmb_node)
            
            for a in range(10):
                optimizer_vae.zero_grad()
                if a < 9:
                    recon_batch_mask, _, _ = self.vae((batch_features_tensor, batch_edge_list, values_mask_new.detach()), beta=0, batch=node_indicator_tensor)
                    recon_batch, mu, logvar = self.vae((batch_features_tensor, batch_edge_list, batch_graph_delta.detach()), beta=1, batch=node_indicator_tensor)

                    
                    aug_edge_weights_delta = recon_batch
                    aug_edge_weights_mask = recon_batch_mask
                    aug_mask = torch.max(aug_edge_weights_mask, aug_edge_weights_delta)

                    loss_vae = self.loss_function(aug_mask, batch_label, mu, logvar, batch_weight_tensor_ori, batch_weight_tensor, batch_norm, aug_edge_weights_mask, values_mask.detach())
                    
                    loss_vae.backward()  
                else:
                    recon_batch_mask, _, _ = self.vae((batch_features_tensor, batch_edge_list, values_mask_new), beta=0, batch=node_indicator_tensor)
                    recon_batch, mu, logvar = self.vae((batch_features_tensor, batch_edge_list, batch_graph_delta), beta=1, batch=node_indicator_tensor)

                    aug_edge_weights_delta = recon_batch
                    aug_edge_weights_mask = recon_batch_mask
                    aug_mask = torch.max(aug_edge_weights_mask, aug_edge_weights_delta)

                    loss_vae = self.loss_function(aug_mask, batch_label, mu, logvar, batch_weight_tensor_ori,batch_weight_tensor, batch_norm, aug_edge_weights_mask, values_mask)
                    loss_vae.backward(retain_graph=True)  
                optimizer_vae.step()
                
            masked_pred, _ = self.model_to_explain(batch_features_tensor,
                                                batch_fully_graphs, 
                                                batch=node_indicator_tensor,
                                                edge_weights=aug_mask.view(-1)) 
                
            target = torch.argmax(original_pred,-1)
            
            loss = self._loss(masked_pred,target, mask, self.reg_coefs)
            loss.backward()
            optimizer.step()
            
        

    def explain(self, index):
        index = int(index)
        feats = self.features[index].clone().detach()
        graph = self.graphs[index].clone().detach()
        all_one_edge_weights = torch.ones(graph.size(1)).to(self.device)
        embeds = self.model_to_explain.embedding(feats, graph,all_one_edge_weights).detach()


        input_expl = self._create_explainer_input(graph, embeds).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()


        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights 
