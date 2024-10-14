import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric.utils as pyg_utils
import torch_sparse

from .neural import NeuralScheduler, ExecPolicyNetwork, make_mlp , HeuristicPolicyNetwork, ResourcePolicyNetwork
from spark_sched_sim.wrappers import DAGNNObsWrapper
from spark_sched_sim import graph_utils


class HyperHeuristicScheduler(NeuralScheduler):

    def __init__(
        self,
        num_executors,
        embed_dim,
        gnn_mlp_kwargs,
        policy_mlp_kwargs,
        state_dict_path=None,
        opt_cls=None,
        opt_kwargs=None,
        max_grad_norm=None,
        num_node_features = 7,
        num_dag_features = 3,
        num_heuristics = 2,
        input_feature = ['num_queue',"glob"],
        list_heuristics= ['FIFO', 'MC'],
        num_resource_heuristics= 3,
        list_resource_heuristics= ['FIFO', 'Fair'],
        resource_allocation = 'Random',
        **kwargs
    ):
        name = "HyperHeuristic"
        # if state_dict_path:
        #     name += f":{state_dict_path}"

        actor = ActorNetwork(
            num_executors,
            num_node_features,
            num_dag_features,
            embed_dim,
            gnn_mlp_kwargs,
            policy_mlp_kwargs,
            num_heuristics,
            list_heuristics,
            input_feature,
            num_resource_heuristics,
            list_resource_heuristics,
            resource_allocation
        )

        obs_wrapper_cls = DAGNNObsWrapper

        super().__init__(
            name,
            actor,
            obs_wrapper_cls,
            num_executors,
            state_dict_path,
            opt_cls,
            opt_kwargs,
            max_grad_norm,
            num_heuristics,
            list_heuristics,
            num_resource_heuristics,
            list_resource_heuristics,
            resource_allocation,
        )


class ActorNetwork(nn.Module):
    def __init__(
        self,
        num_executors,
        num_node_features,
        num_dag_features,
        embed_dim,
        gnn_mlp_kwargs,
        policy_mlp_kwargs,
        num_heuristics,
        list_heuristics,
        input_feature,
        num_resource_heuristics,
        list_resource_heuristics,
        resource_allocation
    ):
        super().__init__()
        self.encoder = EncoderNetwork(num_node_features, embed_dim, gnn_mlp_kwargs)
        self.embedding_model = ComplexHeuristicEmbeddingModel(
            action_size=num_heuristics,
            embedding_dim=embed_dim,
            hidden_dim=64,  # Example hidden dimension size
            dropout=0.1,

        )

        emb_dims = {"resource_heuristic":embed_dim, "heuristic":embed_dim,"node": embed_dim, "dag": embed_dim, "glob": embed_dim}

        self.heuristic_policy_network = HeuristicPolicyNetwork(
            self.embedding_model, num_heuristics, list_heuristics, input_feature, emb_dims, policy_mlp_kwargs
        )

        if resource_allocation == "DNN":
            self.exec_policy_network = ExecPolicyNetwork(
                num_executors, num_dag_features, emb_dims, policy_mlp_kwargs
            )
        elif resource_allocation == "HyperHeuristic":
            self.resource_heuristic_policy_network = ResourcePolicyNetwork(
                self.embedding_model, num_resource_heuristics, list_resource_heuristics,
                num_executors, num_dag_features, emb_dims, policy_mlp_kwargs)

        self._reset_biases()

    def _reset_biases(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                param.data.zero_()


class EncoderNetwork(nn.Module):
    def __init__(self, num_node_features, embed_dim, mlp_kwargs):
        super().__init__()

        self.node_encoder = NodeEncoder(num_node_features, embed_dim, mlp_kwargs)

        self.dag_encoder = DagEncoder(num_node_features, embed_dim, mlp_kwargs)

        self.global_encoder = GlobalEncoder(embed_dim, mlp_kwargs)

    def forward(self, dag_batch):
        """
        Returns:
            a dict of representations at three different levels:
            node, dag, and global.
        """
        h_node = self.node_encoder(dag_batch)

        h_dag = self.dag_encoder(h_node, dag_batch)

        try:
            # batch of obsns
            obs_ptr = dag_batch["obs_ptr"]
            h_glob = self.global_encoder(h_dag, obs_ptr)
        except KeyError:
            # single obs
            h_glob = self.global_encoder(h_dag)

        h_dict = {"node": h_node, "dag": h_dag, "glob": h_glob}

        return h_dict


class NodeEncoder(nn.Module):
    def __init__(self, num_node_features, embed_dim, mlp_kwargs, reverse_flow=True):
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        self.mlp_prep = make_mlp(num_node_features, output_dim=embed_dim, **mlp_kwargs)
        self.mlp_msg = make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)
        self.mlp_update = make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, dag_batch):
        edge_masks = dag_batch["edge_masks"]

        if edge_masks.shape[0] == 0:
            # no message passing to do
            return self._forward_no_mp(dag_batch.x)

        # pre-process the node features into initial representations
        h_init = self.mlp_prep(dag_batch.x)

        # will store all the nodes' representations
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg_utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes
        )

        h[src_node_mask] = self.mlp_update(h_init[src_node_mask])

        edge_masks_it = (
            iter(reversed(edge_masks)) if self.reverse_flow else iter(edge_masks)
        )

        # target-to-source message passing, one level of the dags at a time
        for edge_mask in edge_masks_it:
            edge_index_masked = dag_batch.edge_index[:, edge_mask]
            adj = graph_utils.make_adj(edge_index_masked, num_nodes)

            # nodes sending messages
            src_mask = pyg_utils.index_to_mask(edge_index_masked[self.j], num_nodes)

            # nodes receiving messages
            dst_mask = pyg_utils.index_to_mask(edge_index_masked[self.i], num_nodes)

            msg = torch.zeros_like(h)
            msg[src_mask] = self.mlp_msg(h[src_mask])
            agg = torch_sparse.matmul(adj if self.reverse_flow else adj.t(), msg)
            h[dst_mask] = h_init[dst_mask] + self.mlp_update(agg[dst_mask])

        return h

    def _forward_no_mp(self, x):
        """forward pass without any message passing. Needed whenever
        all the active jobs are almost complete and only have a single
        layer of nodes remaining.
        """
        return self.mlp_prep(x)


class DagEncoder(nn.Module):
    def __init__(self, num_node_features, embed_dim, mlp_kwargs):
        super().__init__()
        input_dim = num_node_features + embed_dim
        self.mlp = make_mlp(input_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, h_node, dag_batch):
        # include original input
        h_node = torch.cat([dag_batch.x, h_node], dim=1)
        h_node_matrix = self.mlp(h_node)  #dim : num_node x output_dim=embed_dim
        h_dag = segment_csr(h_node_matrix, dag_batch.ptr) #sum h_node_matrix value over all nodes in the same dag, dim: num_dag x output_dim
        return h_dag


class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim, mlp_kwargs):
        super().__init__()
        self.mlp = make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, h_dag, obs_ptr=None):
        h_dag = self.mlp(h_dag)

        if obs_ptr is not None:
            # batch of observations
            h_glob = segment_csr(h_dag, obs_ptr)
        else:
            # single observation
            h_glob = h_dag.sum(0).unsqueeze(0)

        return h_glob


class ComplexHeuristicEmbeddingModel(nn.Module):
    def __init__(self, action_size, embedding_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(action_size, embedding_dim)
        #nn.init.uniform_(self.embedding.weight, -0.1, +0.1)
        nn.init.xavier_uniform_(self.embedding.weight)

        #print("*******Init embedding weight:",self.embedding.weight)

        # Additional layers for complexity
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Final linear layer to map back to embedding dimension
        self.fc3 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, action_indices):
        # Lookup embeddings
        x = self.embedding(action_indices)

        # Pass through additional layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        # Map back to original embedding dimension
        x = self.fc3(x)

        return x