import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, LayerNorm, Dropout, Softmax
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, GATConv,TopKPooling
from torch_geometric.nn import TransformerConv,GCNConv, GATConv, SAGEConv, global_mean_pool
import os
import torch_geometric.nn as pyg_nn
os.environ['PYTORCH_GEOMETRIC_CACHE'] = '/tmp/pyg_cache'  # Set to a writable directory
'''
Slightly modified multihead attention for Gamora
'''


from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.nn import GraphConv
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.typing import OptTensor


class SAGPooling(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:

        .. math::
            \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}.

    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConv`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (str or callable, optional): The non-linearity to use.
            (default: :obj:`"tanh"`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: torch.nn.Module = GraphConv,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = 'tanh',
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.gnn = GNN(in_channels, 1, **kwargs)
        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()
        self.select.reset_parameters()


    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        attn: OptTensor = None,
    ) -> Tuple[Tensor, Tensor, OptTensor, OptTensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)
            attn (torch.Tensor, optional): Optional node-level matrix to use
                for computing attention scores instead of using the node
                feature matrix :obj:`x`. (default: :obj:`None`)
    Output:
        node representations
    """

    def __init__(self, in_channels, gnn_emb_dim,num_gnn_layers, heads, num_hops, dropout):
        '''
            in_channels (int): Input feature dimensionality
            gnn_emb_dim (int): Node embedding dimensionality
            heads (int): Number of attention heads
            num_hops (int): Number of hops for GNN
            dropout (float): Dropout rate
        '''
        super(GNN_node, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = gnn_emb_dim
        self.heads = heads
        self.num_hops = num_hops
        self.dropout = dropout
        self.num_layers = num_gnn_layers
        ### List of GN Ns
        self.hoga_conv = HOGA(in_channels, self.hidden_channels, num_gnn_layers, dropout, num_hops + 1, heads)  # HOGA layer
        #self.batch_norm_hoga = torch.nn.BatchNorm1d(self.hidden_channels)

    def forward(self, batched_data):
        h = batched_data

        # Apply HOGA layer
        h_hoga = self.hoga_conv(h)
        # h_hoga = self.batch_norm_hoga(h_hoga)
        # h_hoga = F.relu(h_hoga)

        return h_hoga





class SynthNet(torch.nn.Module):
    def __init__(self, args):
        super(SynthNet, self).__init__()
        super(SynthNet, self).__init__()
        
        # Parameter initialization
        self.in_channels = args.feature_size
        self.num_fc_layers = args.num_fc_layer
        self.num_gnn_layers = args.num_layers
        self.gnn_emb_dim = args.gnn_embedding_dim
        self.hidden_dim = args.hidden_channels
        self.heads = args.heads
        self.num_hops = args.num_hops         
        self.dropout = args.dropout
        self.n_classes_task1 = args.num_n_classes_task1 # Ensure to get number of classes from args
        self.n_classes_task2 = args.num_n_classes_task2
        self.n_classes_task3 = args.num_n_classes_task3
        
        ############### GNN layer############
        self.gnn = GNN(
            self.in_channels, 
            self.gnn_emb_dim, 
            self.num_gnn_layers, 
            self.heads, 
            self.num_hops, 
            self.dropout
        )

        ############### FC layers ############
        self.fc = torch.nn.Linear(self.gnn_emb_dim*2, self.hidden_dim)
        self.batch_norm = torch.nn.BatchNorm1d(self.hidden_dim)
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        # # Linear layers for multi-task outputs
        self.output_layer_task1 = torch.nn.Linear(self.hidden_dim, self.n_classes_task1)  # Task 1
        self.output_layer_task2 = torch.nn.Linear(self.hidden_dim, self.n_classes_task2)  # Task 2
        self.output_layer_task3 = torch.nn.Linear(self.hidden_dim, self.n_classes_task3)  # Task 2

    def forward(self,batch_data):
        # Get graph embeddings from GNN
        graphEmbed = self.gnn(batch_data)
        # print(f"graphEmbed shape: {graphEmbed.shape}")
        

        # Fully connected layer
        x = F.relu(self.fc(graphEmbed))  # Directly use the output from GNN
        x = self.batch_norm(x)
        x = self.dropout_layer(x)

        # Multi-task outputs
        output_task1 = self.output_layer_task1(x)  # Output for Task 1
        output_task2 = self.output_layer_task2(x)  # Output for Task 2
        output_task3 = self.output_layer_task3(x)  # Output for Task 2
        # Apply log_softmax to each output for classification
        # output_task1 = F.log_softmax(output_task1, dim=-1)
        # output_task2 = F.log_softmax(output_task2, dim=-1)
        
        return output_task1, output_task2,output_task3  # Returns outputs for each task
    
    
    
    def reset_parameters(self):
        # Reset parameters of all layers
        self.fc.reset_parameters()
        self.batch_norm.reset_parameters()
        self.output_layer_task1.reset_parameters()
        self.output_layer_task2.reset_parameters()
        self.gnn.reset_parameters()
    # Todo:
    # def reset_parameters(self):
    #     for lin in self.gnn.gnn_node.convs.lins:
    #         lin.reset_parameters()
    #     for gate in self.gates:
    #         gate.reset_parameters()
    #     for li in self.linear:
    #         li.reset_parameters()
class SimpleGNN1_HOGA(torch.nn.Module):
    def __init__(self, args):
        super(SimpleGNN1_HOGA, self).__init__()

        self.in_channels = args.feature_size
        self.hidden_dim = args.hidden_channels
        self.dropout = args.dropout
        self.n_classes_task1 = args.num_n_classes_task1
        self.num_gnn_layers = args.num_layers
        self.num_hops = args.num_hops+1  # Number of hops for HOGA
        self.heads = args.heads  # Number of attention heads
        self.attn_dropout = args.attn_dropout
        self.attn_type = args.attn_type
        self.use_bias = args.use_bias
        self.directed = args.directed
        # Replace SAGEConv layers with HOGA
        self.hoga = HOGA(in_channels=self.in_channels, 
                         hidden_channels=self.hidden_dim, 
                         num_layers=self.num_gnn_layers, 
                         dropout=self.dropout, 
                         num_hops=args.num_hops+1, 
                         heads=self.heads, 
                         directed=self.directed,
                         attn_dropout=self.attn_dropout, 
                         attn_type=self.attn_type, 
                         use_bias=self.use_bias
                         )

        self.norm_emb = BatchNorm1d(self.hidden_dim)
        self.batch_norm_task1 = BatchNorm1d(self.n_classes_task1)
        self.fc_task1 = torch.nn.Linear(self.hidden_dim + 1, self.n_classes_task1)
        self.fc_task2 = torch.nn.Linear(self.hidden_dim + 3, 2)  # Assuming binary classification with additional features
        self.fc_task3_branch1 = torch.nn.Linear(self.hidden_dim + 5, 4)  # For classes 1-4
        self.fc_task3_branch2 = torch.nn.Linear(self.hidden_dim + 5, 5)  # For classes 5-9
        self.norm_combined = BatchNorm1d(self.hidden_dim + 5)
        self.reset_parameters()

    def reset_parameters(self):
        self.hoga.reset_parameters()
        self.norm_emb.reset_parameters()
        self.norm_combined.reset_parameters()
        self.fc_task1.reset_parameters()
        self.fc_task2.reset_parameters()
        self.fc_task3_branch1.reset_parameters()
        self.fc_task3_branch2.reset_parameters()

    def forward(self, batch_data, task):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        graph_features = batch_data.graph_feature
        
        x = x.float()
        #x = x[:, :-1]  # Assuming the last column is not needed in input features
        # print("x shape:", x.shape)
        # Pass through the HOGA layer
        x = self.hoga(x)  # This will process the input through HOGA

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        x = self.norm_emb(x)
        
        lev_lsb = graph_features[:, -1].unsqueeze(1)  # 
        networkx_features = graph_features[:, -3:]  # 

        x_task1 = torch.cat([x, lev_lsb], dim=1)  # [batch_size, hidden_dim + 1]
        x_task2 = torch.cat([x, networkx_features], dim=1)  # [batch_size, hidden_dim + 3]

        output_task1 = self.fc_task1(x_task1)
        output_task1 = F.relu(output_task1)
        output_task1 = self.batch_norm_task1(output_task1)
        output_task1 = F.dropout(output_task1, p=self.dropout, training=self.training)

        output_task2 = self.fc_task2(x_task2)
        output_task2 = F.relu(output_task2)
        output_task2 = F.dropout(output_task2, p=self.dropout, training=self.training)

        task2_prediction = output_task2.argmax(dim=1)  #

        weighted_feature_1 = graph_features[:, 2].unsqueeze(1)  #
        weighted_feature_2 = graph_features[:, 1].unsqueeze(1)  # 
        x_task3 = torch.cat([x, weighted_feature_1, weighted_feature_2, networkx_features], dim=1)  # [batch_size, hidden_dim + 2]
        x_task3 = self.norm_combined(x_task3)

        output_task3_branch1 = self.fc_task3_branch1(x_task3)  # [batch_size, 4]
        output_task3_branch2 = self.fc_task3_branch2(x_task3)  # [batch_size, 5]

        output_task3 = torch.zeros((x.size(0), 9)).to(x.device)  # [batch_size, 9]

        output_task3[task2_prediction == 0, :4] = output_task3_branch1[task2_prediction == 0]
        output_task3[task2_prediction == 1, 4:] = output_task3_branch2[task2_prediction == 1]
        
        return output_task1, output_task2, output_task3    
class SimpleGNN1(torch.nn.Module):
    def __init__(self, args):
        super(SimpleGNN1, self).__init__()

        self.in_channels = args.feature_size 
        
        #self.in_channels = args.feature_size 
        # self.gnn_emb_dim = args.gnn_embedding_dim
        self.hidden_dim = args.hidden_channels
        self.dropout = args.dropout
        self.n_classes_task1 = args.num_n_classes_task1
        self.num_gnn_layers = args.num_layers
        self.graph_feature_size = args.graph_feature_size
        self.convs = torch.nn.ModuleList()

        for i in range(self.num_gnn_layers - 1):
            in_channels = self.in_channels if i == 0 else self.hidden_dim
            self.convs.append(GATConv(in_channels, self.hidden_dim))
        # self.gat = GATConv(self.hidden_dim, self.hidden_dim)
        # for i in range(self.num_gnn_layers):
        #     in_channels = self.in_channels if i == 0 else self.hidden_dim
        #     self.convs.append(SAGEConv(in_channels, self.hidden_dim))
        
        self.norm_emb = BatchNorm1d(self.hidden_dim)
        self.batch_norm_task1 = BatchNorm1d(self.n_classes_task1)
        self.fc_task1 = torch.nn.Linear(self.hidden_dim + 1, self.n_classes_task1)
        self.fc_task2 = torch.nn.Linear(self.hidden_dim + 3, 2)  # Assuming binary classification with additional features
        self.fc_task3_branch1 = torch.nn.Linear(self.hidden_dim + 5, 4)  # For classes 1-4
        self.fc_task3_branch2 = torch.nn.Linear(self.hidden_dim + 5, 5)  # For classes 5-9
        self.norm_combined = BatchNorm1d(self.hidden_dim + 5)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # self.gat.reset_parameters()
        self.norm_emb.reset_parameters()
        self.norm_combined.reset_parameters()
        self.fc_task1.reset_parameters()
        self.fc_task2.reset_parameters()
        self.fc_task3_branch1.reset_parameters()
        self.fc_task3_branch2.reset_parameters()

    def forward(self, batch_data, task):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        graph_features = batch_data.graph_feature
         
        x = x.float()
        #x = x[:, :-1]  

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x = self.gat(x, edge_index)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        x = self.norm_emb(x)
        
        lev_lsb = graph_features[:, -1].unsqueeze(1)  # 
        networkx_features = graph_features[:, -3:]  # 

        x_task1 = torch.cat([x, lev_lsb], dim=1)  # [batch_size, hidden_dim + 1]
        x_task2 = torch.cat([x, networkx_features], dim=1)  # [batch_size, hidden_dim + 3]

        output_task1 = self.fc_task1(x_task1)
        output_task1 = F.relu(output_task1)
        output_task1 = self.batch_norm_task1(output_task1)
        output_task1 = F.dropout(output_task1, p=self.dropout, training=self.training)

        output_task2 = self.fc_task2(x_task2)
        output_task2 = F.relu(output_task2)
        output_task2 = F.dropout(output_task2, p=self.dropout, training=self.training)

        task2_prediction = output_task2.argmax(dim=1)  #

        weighted_feature_1 = graph_features[:, 2].unsqueeze(1)  #
        weighted_feature_2 = graph_features[:, 1].unsqueeze(1)  # 
        x_task3 = torch.cat([x, weighted_feature_1, weighted_feature_2,networkx_features], dim=1)  # [batch_size, hidden_dim + 2]
        x_task3 = self.norm_combined(x_task3)

        output_task3_branch1 = self.fc_task3_branch1(x_task3)  # [batch_size, 4]
        output_task3_branch2 = self.fc_task3_branch2(x_task3)  # [batch_size, 5]

        output_task3 = torch.zeros((x.size(0), 9)).to(x.device)  # [batch_size, 9]

        output_task3[task2_prediction == 0, :4] = output_task3_branch1[task2_prediction == 0]
       
        output_task3[task2_prediction == 1, 4:] = output_task3_branch2[task2_prediction == 1]
        
        
        return output_task1, output_task2, output_task3


class BiDirectionalGraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super(BiDirectionalGraphSAGE, self).__init__()

        self.in_channels = args.feature_size
        self.hidden_dim = args.hidden_channels
        self.dropout = args.dropout
        self.n_classes_task1 = args.num_n_classes_task1
        self.num_gnn_layers = args.num_layers
        self.graph_feature_size = args.graph_feature_size

        self.pre_sage_convs = torch.nn.ModuleList()
        self.suc_sage_convs = torch.nn.ModuleList()

        for i in range(self.num_gnn_layers - 1):
            in_channels = self.in_channels if i == 0 else self.hidden_dim
            self.pre_sage_convs.append(SAGEConv(in_channels, self.hidden_dim))
            self.suc_sage_convs.append(SAGEConv(in_channels, self.hidden_dim))

        self.norm_emb = BatchNorm1d(self.hidden_dim * 2)
        self.batch_norm_task1 = BatchNorm1d(self.n_classes_task1)
        self.fc_task1 = torch.nn.Linear(self.hidden_dim * 2 + 1, self.n_classes_task1)
        self.fc_task2 = torch.nn.Linear(self.hidden_dim * 2 + 3, 2)
        self.fc_task3_branch1 = torch.nn.Linear(self.hidden_dim * 2 + 5, 4)
        self.fc_task3_branch2 = torch.nn.Linear(self.hidden_dim * 2 + 5, 5)
        self.norm_combined = BatchNorm1d(self.hidden_dim * 2 + 5)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.pre_sage_convs:
            conv.reset_parameters()
        for conv in self.suc_sage_convs:
            conv.reset_parameters()
        self.norm_emb.reset_parameters()
        self.norm_combined.reset_parameters()
        self.batch_norm_task1.reset_parameters()
        self.fc_task1.reset_parameters()
        self.fc_task2.reset_parameters()
        self.fc_task3_branch1.reset_parameters()
        self.fc_task3_branch2.reset_parameters()

    def forward(self, batch_data, task):
        x, edge_index, edge_direction, batch = batch_data.x, batch_data.edge_index, batch_data.edge_direction, batch_data.batch
        graph_features = batch_data.graph_feature
        x = x.float()
        pre_edge_index = edge_index[:, edge_direction == 0]
        suc_edge_index = edge_index[:, edge_direction == 1]
        pre_x = x
        suc_x = x
    
        for idx, (pre_conv, suc_conv) in enumerate(zip(self.pre_sage_convs, self.suc_sage_convs)):
            pre_x = F.relu(pre_conv(pre_x, pre_edge_index))
            pre_x = F.dropout(pre_x, p=self.dropout, training=self.training)
            suc_x = F.relu(suc_conv(suc_x, suc_edge_index))
            suc_x = F.dropout(suc_x, p=self.dropout, training=self.training)
    
    
        combined_x = torch.cat([pre_x, suc_x], dim=1)  # [num_nodes, hidden_dim * 2]
    
        combined_x = global_mean_pool(combined_x, batch)  # [num_graphs, hidden_dim * 2]
        
        combined_x = self.norm_emb(combined_x)
    

    
        if graph_features.dim() == 2 and graph_features.size(0) == x.size(0):
            graph_features = global_mean_pool(graph_features, batch)
        elif graph_features.dim() == 2 and graph_features.size(0) == combined_x.size(0):

            pass
        else:
            print('Warning: graph_features shape may be incorrect!')
    
        lev_lsb = graph_features[:, -1].unsqueeze(1)  # 
        networkx_features = graph_features[:, -3:]  # 
    
        x_task1 = torch.cat([combined_x, lev_lsb], dim=1)  # [batch_size, hidden_dim * 2 + 1]

        x_task2 = torch.cat([combined_x, networkx_features], dim=1)  # [batch_size, hidden_dim * 2 + 3]

        output_task1 = self.fc_task1(x_task1)
        output_task1 = F.relu(output_task1)
        output_task1 = self.batch_norm_task1(output_task1)
        output_task1 = F.dropout(output_task1, p=self.dropout, training=self.training)

        output_task2 = self.fc_task2(x_task2)
        output_task2 = F.relu(output_task2)
        output_task2 = F.dropout(output_task2, p=self.dropout, training=self.training)

    
        task2_prediction = output_task2.argmax(dim=1)  # [batch_size]

    
        weighted_feature_1 = graph_features[:, 2].unsqueeze(1)  #
        weighted_feature_2 = graph_features[:, 1].unsqueeze(1)  # 
        x_task3 = torch.cat([combined_x, weighted_feature_1, weighted_feature_2, networkx_features], dim=1)  # [batch_size, hidden_dim * 2 + 5]
        x_task3 = self.norm_combined(x_task3)
    
        output_task3_branch1 = self.fc_task3_branch1(x_task3)  # [batch_size, 4]
        output_task3_branch2 = self.fc_task3_branch2(x_task3)  # [batch_size, 5]

    
        output_task3 = torch.zeros((combined_x.size(0), 9)).to(x.device)  # [batch_size, 9]
    
        mask0 = task2_prediction == 0
        mask1 = task2_prediction == 1
    
        output_task3[mask0, :4] = output_task3_branch1[mask0]
        output_task3[mask1, 4:] = output_task3_branch2[mask1]
    
        return output_task1, output_task2, output_task3

class SimpleGraphTransformer(torch.nn.Module):
    def __init__(self, args):
        super(SimpleGraphTransformer, self).__init__()

        self.in_channels = args.feature_size 
        self.hidden_dim = args.hidden_channels
        self.dropout = args.dropout
        self.n_classes_task1 = args.num_n_classes_task1
        self.num_gnn_layers = args.num_layers
        self.graph_feature_size = args.graph_feature_size

        self.convs = torch.nn.ModuleList()

        self.num_heads = 4
        for i in range(self.num_gnn_layers - 1):
            in_channels = self.in_channels if i == 0 else self.hidden_dim
            self.convs.append(TransformerConv(
                in_channels=in_channels,
                out_channels=self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                concat=True,
                dropout=self.dropout
            ))
        
        self.norm_emb = BatchNorm1d(self.hidden_dim)
        self.batch_norm_task1 = BatchNorm1d(self.n_classes_task1)
        self.fc_task1 = torch.nn.Linear(self.hidden_dim + 1, self.n_classes_task1)
        self.fc_task2 = torch.nn.Linear(self.hidden_dim + 3, 2)  # Assuming binary classification with additional features
        self.fc_task3_branch1 = torch.nn.Linear(self.hidden_dim + 5, 4)  # For classes 1-4
        self.fc_task3_branch2 = torch.nn.Linear(self.hidden_dim + 5, 5)  # For classes 5-9
        self.norm_combined = BatchNorm1d(self.hidden_dim + 5)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.norm_emb.reset_parameters()
        self.norm_combined.reset_parameters()
        self.fc_task1.reset_parameters()
        self.fc_task2.reset_parameters()
        self.fc_task3_branch1.reset_parameters()
        self.fc_task3_branch2.reset_parameters()

    def forward(self, batch_data, task):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        graph_features = batch_data.graph_feature
         
        x = x.float()
        #x = x[:, :-1]  

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        x = self.norm_emb(x)
        
        lev_lsb = graph_features[:, -1].unsqueeze(1)  # 
        networkx_features = graph_features[:, -3:]  # 

        x_task1 = torch.cat([x, lev_lsb], dim=1)  # [batch_size, hidden_dim + 1]
        x_task2 = torch.cat([x, networkx_features], dim=1)  # [batch_size, hidden_dim + 3]

        output_task1 = self.fc_task1(x_task1)
        output_task1 = F.relu(output_task1)
        output_task1 = self.batch_norm_task1(output_task1)
        output_task1 = F.dropout(output_task1, p=self.dropout, training=self.training)

        output_task2 = self.fc_task2(x_task2)
        output_task2 = F.relu(output_task2)
        output_task2 = F.dropout(output_task2, p=self.dropout, training=self.training)

        task2_prediction = output_task2.argmax(dim=1)  #
        
        weighted_feature_1 = graph_features[:, 2].unsqueeze(1)  #
        weighted_feature_2 = graph_features[:, 1].unsqueeze(1)  # 
        x_task3 = torch.cat([x, weighted_feature_1, weighted_feature_2, networkx_features], dim=1)  # [batch_size, hidden_dim + 2]
        x_task3 = self.norm_combined(x_task3)

        output_task3_branch1 = self.fc_task3_branch1(x_task3)  # [batch_size, 4]
        output_task3_branch2 = self.fc_task3_branch2(x_task3)  # [batch_size, 5]

        output_task3 = torch.zeros((x.size(0), 9)).to(x.device)  # [batch_size, 9]

        output_task3[task2_prediction == 0, :4] = output_task3_branch1[task2_prediction == 0]
       
        output_task3[task2_prediction == 1, 4:] = output_task3_branch2[task2_prediction == 1]
        
        return output_task1, output_task2, output_task3
class SimpleGNN_SAGPool(torch.nn.Module):
    def __init__(self, args):
        super(SimpleGNN_SAGPool, self).__init__()

        self.in_channels = args.feature_size 
        self.gnn_emb_dim = args.gnn_embedding_dim
        self.hidden_dim = args.hidden_channels
        self.dropout = args.dropout
        self.pooling_ratio=args.pooling_ratio
        self.n_classes_task1 = args.num_n_classes_task1
        self.num_gnn_layers = args.num_layers
        self.graph_feature_size = args.graph_feature_size

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(self.num_gnn_layers):
            conv = GCNConv(self.in_channels if i == 0 else self.gnn_emb_dim, self.gnn_emb_dim)
            self.convs.append(conv)
            pool = SAGPooling(self.gnn_emb_dim, ratio=self.pooling_ratio)
            self.pools.append(pool)

        self.gat = GATConv(self.gnn_emb_dim, self.hidden_dim)

        self.norm_emb = BatchNorm1d(self.hidden_dim)
        self.batch_norm_task1 = BatchNorm1d(self.n_classes_task1)

        self.fc_task1 = torch.nn.Linear(self.hidden_dim + 1, self.n_classes_task1)

        self.fc_task2 = torch.nn.Linear(self.hidden_dim + 3, 2)

        self.fc_task3_branch1 = torch.nn.Linear(self.hidden_dim + 2, 4)
        self.fc_task3_branch2 = torch.nn.Linear(self.hidden_dim + 2, 5)

        self.norm_combined = BatchNorm1d(self.hidden_dim + 2)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.gat.reset_parameters()
        self.norm_emb.reset_parameters()
        self.norm_combined.reset_parameters()
        self.fc_task1.reset_parameters()
        self.fc_task2.reset_parameters()
        self.fc_task3_branch1.reset_parameters()
        self.fc_task3_branch2.reset_parameters()

    def forward(self, batch_data, task=None):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch
        graph_features = batch_data.graph_feature

        x = x.float()

        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, perm, score = pool(x, edge_index, None, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]

        x = self.norm_emb(x)  # [num_graphs, hidden_dim]

        lev_lsb = graph_features[:, -1].unsqueeze(1)
        networkx_features = graph_features[:, -3:]

        x_task1 = torch.cat([x, lev_lsb], dim=1)  # [batch_size, hidden_dim + 1]
        x_task2 = torch.cat([x, networkx_features], dim=1)  # [batch_size, hidden_dim + 3]

        output_task1 = self.fc_task1(x_task1)
        output_task1 = F.relu(output_task1)
        output_task1 = self.batch_norm_task1(output_task1)
        output_task1 = F.dropout(output_task1, p=self.dropout, training=self.training)

        output_task2 = self.fc_task2(x_task2)
        output_task2 = F.relu(output_task2)
        output_task2 = F.dropout(output_task2, p=self.dropout, training=self.training)

        task2_prediction = output_task2.argmax(dim=1)

        weighted_feature_1 = graph_features[:, 2].unsqueeze(1)
        weighted_feature_2 = graph_features[:, 1].unsqueeze(1)
        x_task3 = torch.cat([x, weighted_feature_1, weighted_feature_2], dim=1)  # [batch_size, hidden_dim + 2]
        x_task3 = self.norm_combined(x_task3)

        output_task3_branch1 = self.fc_task3_branch1(x_task3)  # [batch_size, 4]
        output_task3_branch2 = self.fc_task3_branch2(x_task3)  # [batch_size, 5]

        output_task3 = torch.zeros((x.size(0), 9)).to(x.device)  # [batch_size, 9]

        mask_branch1 = (task2_prediction == 0)
        output_task3[mask_branch1, :4] = output_task3_branch1[mask_branch1]

        mask_branch2 = (task2_prediction == 1)
        output_task3[mask_branch2, 4:] = output_task3_branch2[mask_branch2]

        return output_task1, output_task2, output_task3