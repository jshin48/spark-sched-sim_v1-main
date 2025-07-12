from collections.abc import Iterable
from typing import Any
from torch import Tensor

import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric as pyg
import torch_sparse

from .utils import make_mlp

from ..scheduler import TrainableScheduler
from .env_wrapper import DecimaEnvWrapper
from . import utils



class HyperHeuristicScheduler(TrainableScheduler):
    """Original Decima architecture, which uses asynchronous message passing
    as in DAGNN.
    Paper: https://dl.acm.org/doi/abs/10.1145/3341302.3342080
    """

    def __init__(
        self,
        num_executors: int,
        embed_dim: int,
        gnn_mlp_kwargs: dict[str, Any],
        policy_mlp_kwargs: dict[str, Any],
        state_dict_path: str | None = None,
        opt_cls: str | None = None,
        opt_kwargs: dict[str, Any] | None = None,
        max_grad_norm: float | None = None,
        num_node_features: int = 7,
        num_dag_features: int = 3,
        num_heuristics: int = 2,
        input_feature = ['num_queue', "glob"],
        list_heuristics = ['FIFO', 'MC'],
        num_resource_heuristics = 3,
        list_resource_heuristics = ['FIFO', 'Fair'],
        resource_allocation='Random',
        **kwargs,
    ):
        super().__init__()

        self.name = "HyperHeuristic"
        self.env_wrapper_cls = DecimaEnvWrapper
        self.max_grad_norm = max_grad_norm
        self.num_executors = num_executors
        self.heuristics_count = [0 for i in range(self.num_heuristics)]

        self.encoder = EncoderNetwork(num_node_features, embed_dim, gnn_mlp_kwargs)
        self.embedding_model = ComplexHeuristicEmbeddingModel(
            action_size=num_heuristics,
            embedding_dim=embed_dim,
            hidden_dim=64,  # Example hidden dimension size
            dropout=0.1,
        )

        emb_dims = {"resource_heuristic": embed_dim, "heuristic": embed_dim,
                    "node": embed_dim, "dag": embed_dim, "glob": embed_dim}

        self.heuristic_policy_network = HeuristicPolicyNetwork(
            self.embedding_model, num_heuristics, list_heuristics,
            input_feature, emb_dims, policy_mlp_kwargs
        )

        self._reset_biases()

        if state_dict_path:
            self.name += f":{state_dict_path}"
            self.load_state_dict(torch.load(state_dict_path))

        if opt_cls:
            self.optim = getattr(torch.optim, opt_cls)(
                self.parameters(), **(opt_kwargs or {})
            )

    def _reset_biases(self) -> None:
        for name, param in self.named_parameters():
            if "bias" in name:
                param.data.zero_()

    @torch.no_grad()
    def schedule(self, obs: dict) -> tuple[dict, dict]:
        dag_batch = utils.obs_to_pyg(obs)
        dag_batch.to(self.device, non_blocking=True)

        # 1. prepare feature dictionary for heuristic policy network
        feature_dict = self.prepare_feature_dict(self, dag_batch)

        # 2. select a heuristic
        heuristic_score = self.heuristic_policy_network(feature_dict)
        heuristic_idx, lgprob = utils.sample(heuristic_score)

        # if heuristic_idx == 0:
        #     scheduler = WscptScheduler(self.num_executors, self.resource_allocation)
        # elif heuristic_idx == 1:
        #     scheduler = McScheduler(self.num_executors, self.resource_allocation)
        # elif heuristic_idx == 2:
        #     scheduler = SjfScheduler(self.num_executors, self.resource_allocation)
        # elif heuristic_idx == 3:
        #     scheduler = FifoScheduler(self.num_executors, self.resource_allocation)
        # else:
        #     sys.exit("Heuristic idx is not matched to any scheduler")
        # self.heuristics_count[heuristic_idx] += 1
        # action = scheduler(obs)
        # stage_idx = action['stage_idx']

        ## 3. retrieve index of selected stage's job
        # try:
        #     stage_idx_glob = pyg.utils.mask_to_index(stage_mask)[stage_idx]
        # except:
        #     print(obs["dag_ptr"])
        # job_idx = stage_to_job_map[stage_idx_glob].item()

        # 4. select the number of executors to add to that stage, conditioned
        # on that stage's job & Calculate lgprob
        # if self.resource_allocation == "HyperHeuristic":
        #     resource_heuristic_score = self.actor.resource_heuristic_policy_network(dag_batch, h_dict, job_idx)
        #     resource_heuristic_idx, resource_lgprob = utils.sample(resource_heuristic_score)
        #     self.resource_heuristics_count[resource_heuristic_idx ] += 1
        #     num_exec = ResourceHeuristics(resource_heuristic_idx,obs,job_idx)
        #     lgprob = lgprob + resource_lgprob
        # else:
        #     resource_heuristic_idx = -1
        #     if self.resource_allocation == 'Random':
        #         num_exec = random.randint(0, obs["num_committable_execs"])
        #     elif self.resource_allocation == 'DNN':
        #         exec_scores = self.actor.exec_policy_network(dag_batch, h_dict, job_idx)
        #         num_exec, exec_lgprob = utils.sample(exec_scores)
        #         lgprob = lgprob + exec_lgprob
        #     elif self.resource_allocation == 'DRA':
        #         num_exec = action['num_exec']
        #     else:
        #         sys.exit("Check -resource allocation parameter.")

        action = {'heuristic_idx': heuristic_idx}

        return action, {"lgprob": lgprob}

    # Prepare a feature dictionary for the heuristic policy network
    def prepare_feature_dict(self, dag_batch):
        feature_dict = dict()

        if "avg_glob" in self.input_feature:
            h_dict = self.actor.encoder(dag_batch)
            feature_dict["glob"] = h_dict['glob'].mean(dim=0, keepdim=True)

        stage_mask = dag_batch['stage_mask']
        if "num_queue" in self.input_feature:
            feature_dict['num_queue'] = torch.sum(stage_mask)

        if "cpt_mean" in self.input_feature or "cpt_var" in self.input_feature:
            stage_cpt = dag_batch.x[:, 5][stage_mask]
            if "cpt_mean" in self.input_feature:
                feature_dict["cpt_mean"] = torch.mean(stage_cpt)
            if "cpt_var" in self.input_feature:
                feature_dict["cpt_var"] = torch.std(stage_cpt)

        if "children_mean" in self.input_feature or "children_var" in self.input_feature:
            stage_children = dag_batch.x[:, 6][stage_mask]
            if "children_mean" in self.input_feature:
                feature_dict["children_mean"] = torch.mean(stage_children)
            if "children_var" in self.input_feature:
                feature_dict["children_var"] = torch.std(stage_children)

        return feature_dict

    def evaluate_actions(
        self, obsns: Iterable[dict], actions: Iterable[tuple]
    ) -> dict[str, Tensor]:
        dag_batch = utils.collate_obsns(obsns)
        heuristic_selections = torch.tensor(actions)
        # obs_ptr = dag_batch["obs_ptr"]

        # re-feed all the observations into the model with grads enabled
        dag_batch.to(self.device)
        feature_dict = self.prepare_feature_dict(self, dag_batch)
        heuristic_score = self.heuristic_policy_network(feature_dict)
        heuristic_lgprobs, heuristic_entropies = utils.evaluate(
            heuristic_score.cpu(), torch.tensor([self.num_heuristics] * len(actions)),
            heuristic_selections)
        action_lgprobs = heuristic_lgprobs
        action_entropies = heuristic_entropies
        # #evaluate resource allocation model
        # if self.resource_allocation == "DNN":
        #     exec_scores = self.actor.exec_policy_network(dag_batch, h_dict, job_indices)
        #     exec_lgprobs, exec_entropies = utils.evaluate(
        #         exec_scores.cpu(), num_exec_acts[job_indices], exec_selections)
        #     action_lgprobs += exec_lgprobs
        #     action_entropies += exec_entropies
        # elif self.resource_allocation == "HyperHeuristic":
        #     resource_heuristic_scores = self.actor.resource_heuristic_policy_network(dag_batch, h_dict, job_indices)
        #     resource_heuristic_lgprobs, resource_heuristic_entropies = utils.evaluate(
        #         resource_heuristic_scores.cpu(), torch.tensor([self.num_resource_heuristics] * len(job_indices)), resource_heuristic_selections)
        #     action_lgprobs += resource_heuristic_lgprobs
        #     action_entropies += resource_heuristic_entropies

        # Normalize entropies
        action_entropies /= (self.num_executors * torch.tensor([self.num_heuristics])).log()
        return {"lgprobs": action_lgprobs, "entropies": action_entropies}


class EncoderNetwork(nn.Module):
    def __init__(
        self, num_node_features: int, embed_dim: int, mlp_kwargs: dict[str, Any]
    ) -> None:
        super().__init__()

        self.node_encoder = NodeEncoder(num_node_features, embed_dim, mlp_kwargs)
        self.dag_encoder = DagEncoder(num_node_features, embed_dim, mlp_kwargs)
        self.global_encoder = GlobalEncoder(embed_dim, mlp_kwargs)

    def forward(self, dag_batch: pyg.data.Batch) -> dict[str, Tensor]:
        """
        Returns:
            a dict of representations at three different levels:
            node, dag, and global.
        """
        h_node = self.node_encoder(dag_batch)

        h_dag = self.dag_encoder(h_node, dag_batch)

        if "obs_ptr" in dag_batch:
            # batch of obsns
            obs_ptr = dag_batch["obs_ptr"]
            h_glob = self.global_encoder(h_dag, obs_ptr)
        else:
            # single obs
            h_glob = self.global_encoder(h_dag)

        return {"node": h_node, "dag": h_dag, "glob": h_glob}


class NodeEncoder(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        embed_dim: int,
        mlp_kwargs: dict[str, Any],
        reverse_flow: bool = True,
    ) -> None:
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        self.mlp_prep = utils.make_mlp(
            num_node_features, output_dim=embed_dim, **mlp_kwargs
        )
        self.mlp_msg = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)
        self.mlp_update = utils.make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, dag_batch: pyg.data.Batch) -> Tensor:
        """returns a tensor of shape [num_nodes, embed_dim]"""

        edge_masks = dag_batch["edge_masks"]

        if edge_masks.shape[0] == 0:
            # no message passing to do
            return self._forward_no_mp(dag_batch.x)

        # pre-process the node features into initial representations
        h_init = self.mlp_prep(dag_batch.x)

        # will store all the nodes' representations
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg.utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes
        )

        h[src_node_mask] = self.mlp_update(h_init[src_node_mask])

        edge_masks_it = (
            iter(reversed(edge_masks)) if self.reverse_flow else iter(edge_masks)
        )

        # target-to-source message passing, one level of the dags at a time
        for edge_mask in edge_masks_it:
            edge_index_masked = dag_batch.edge_index[:, edge_mask]
            adj = utils.make_adj(edge_index_masked, num_nodes)

            # nodes sending messages
            src_mask = pyg.utils.index_to_mask(edge_index_masked[self.j], num_nodes)

            # nodes receiving messages
            dst_mask = pyg.utils.index_to_mask(edge_index_masked[self.i], num_nodes)

            msg = torch.zeros_like(h)
            msg[src_mask] = self.mlp_msg(h[src_mask])
            agg = torch_sparse.matmul(adj if self.reverse_flow else adj.t(), msg)
            h[dst_mask] = h_init[dst_mask] + self.mlp_update(agg[dst_mask])

        return h

    def _forward_no_mp(self, x: Tensor) -> Tensor:
        """forward pass without any message passing. Needed whenever
        all the active jobs are almost complete and only have a single
        layer of nodes remaining.
        """
        return self.mlp_prep(x)


class DagEncoder(nn.Module):
    def __init__(
        self, num_node_features: int, embed_dim: int, mlp_kwargs: dict[str, Any]
    ) -> None:
        super().__init__()
        input_dim = num_node_features + embed_dim
        self.mlp = utils.make_mlp(input_dim, output_dim=embed_dim, **mlp_kwargs)

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

class HeuristicPolicyNetwork(nn.Module):
    def __init__(self, embedding_model, num_heuristics, list_heuristics, input_feature, emb_dims, mlp_kwargs):
        super().__init__()
        self.num_heuristics = num_heuristics
        self.list_heuristics = list_heuristics
        self.input_feature = input_feature
        self.embedding_model = embedding_model

        self.total_feature_list = ["avg_glob","num_queue","cpt_mean", "cpt_var", "children_mean", "children_var"]
        total_dim_list = [emb_dims['glob'], 1, 1, 1, 1, 1]
        # Rearrange input_features to match the dimensions
        feature_in_use = [feature in self.input_feature for feature in self.total_feature_list]
        input_dim = sum(dim for dim, use in zip(total_dim_list, feature_in_use) if use) + emb_dims['heuristic']

        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, dag_batch, feature_dict):
        features = []  # List to store processed features

        for key, value in feature_dict.items():
            if key == 'glob':
                # Repeat 'glob' to match the number of heuristics
                features.append(value.repeat_interleave(self.num_heuristics, dim=0))
            else:
                # Unsqueeze and repeat other features
                features.append(value.unsqueeze(1).repeat_interleave(self.num_heuristics, dim=0))

        # Concatenate all features into a single tensor
        feature_inputs = torch.cat(features, dim=1)

        # Process heuristic actions
        action_indices = torch.arange(self.num_heuristics)
        heuristic_actions = self.embedding_model(action_indices)
        heuristic_actions = heuristic_actions.repeat(feature_inputs.shape[0] // self.num_heuristics, 1)

        # Concatenate features and heuristic actions
        state_inputs = torch.cat([feature_inputs, heuristic_actions], dim=1)

        # Compute heuristic scores
        heuristic_scores = self.mlp_score(state_inputs).squeeze(-1)

        return heuristic_scores
    # #def forward(self, dag_batch, feature_dict):
    #     stage_mask = dag_batch["stage_mask"]
    #     batch_size = feature_dict['glob'].shape[0]
    #
    #     features = []  # Initialize a list to store features
    #     for key, value in feature_dict.items():
    #         if key == 'glob':
    #             features.append(feature_dict['glob'].repeat_interleave(self.num_heuristics, dim=0))
    #         elif key == 'num_queue':
    #             # Process 'num_queue' and append to the list
    #             features.append(value.unsqueeze(1).repeat_interleave(self.num_heuristics, dim=0))
    #         else:
    #             # Process other features and append to the list
    #             feature_value = value.repeat_interleave(self.num_heuristics, dim=0)
    #             if key == "cpt_mean":
    #                 features.append(feature_value.unsqueeze(1))
    #             elif key == "cpt_var":
    #                 features.append(feature_value.unsqueeze(1))
    #             elif key == "children_mean":
    #                 features.append(feature_value.unsqueeze(1))
    #             elif key == "children_var":
    #                 features.append(feature_value.unsqueeze(1))
    #
    #     # Concatenate all features into a single tensor
    #     feature_inputs = torch.cat(features, dim=1)
    #     action_indices = torch.LongTensor(range(self.num_heuristics))
    #     heuristic_actions = self.embedding_model(action_indices)
    #     heuristic_actions = heuristic_actions.repeat(feature_inputs.shape[0], 1).flatten().unsqueeze(1)
    #     state_inputs = torch.cat([feature_inputs , heuristic_actions], dim=1)
    #     heuristic_scores = self.mlp_score(state_inputs).squeeze(-1)
    #
    #     for key, value in feature_dict.items():
    #         if key == 'glob':
    #             features.append(feature_dict['glob'].repeat_interleave(self.num_heuristics, dim=0))
    #         elif key == 'num_queue':
    #             # Process 'num_queue' and append to the list
    #             features.append(value.unsqueeze(1).repeat_interleave(self.num_heuristics, dim=0))
    #         else:
    #             # Process other features and append to the list
    #             feature_value = value.repeat_interleave(self.num_heuristics, dim=0)
    #             if key == "cpt_mean":
    #                 features.append(feature_value.unsqueeze(1))
    #             elif key == "cpt_var":
    #                 features.append(feature_value.unsqueeze(1))
    #             elif key == "children_mean":
    #                 features.append(feature_value.unsqueeze(1))
    #             elif key == "children_var":
    #                 features.append(feature_value.unsqueeze(1))
    #
    #     # Concatenate all features into a single tensor
    #     feature_inputs = torch.cat(features, dim=1)
    #     action_indices = torch.LongTensor(range(self.num_heuristics))
    #     heuristic_actions = self.embedding_model(action_indices)
    #     heuristic_actions = heuristic_actions.repeat(feature_inputs.shape[0], 1).flatten().unsqueeze(1)
    #     state_inputs = torch.cat([feature_inputs , heuristic_actions], dim=1)
    #     heuristic_scores = self.mlp_score(state_inputs).squeeze(-1)
    #
    #     return heuristic_scores

class ResourcePolicyNetwork(nn.Module):
    def __init__(self, embedding_model, num_resource_heuristics, list_resource_heuristics,
                 num_executors, num_dag_features, emb_dims, mlp_kwargs):
        super().__init__()
        self.num_executors = num_executors
        self.num_dag_features = num_dag_features
        self.num_resource_heuristics = num_resource_heuristics
        self.list_resource_heuristics = list_resource_heuristics
        self.embedding_model = embedding_model

        input_dim = num_dag_features + emb_dims["dag"] + emb_dims["glob"] + emb_dims["heuristic"]
        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, dag_batch, feature_dict, job_indices):
        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, : self.num_dag_features]
        x_dag = x_dag[job_indices]
        h_dag = feature_dict["dag"][job_indices]

        try:
            # batch of obsns
            num_exec_acts = dag_batch["num_exec_acts"][job_indices]
        except KeyError:
            # single obs
            x_dag = x_dag.unsqueeze(0)
            h_dag = h_dag.unsqueeze(0)

        # residual connections to original features
        x_h_dag = torch.cat([x_dag, h_dag], dim=1)
        x_h_dag_rpt = x_h_dag.repeat_interleave(self.num_resource_heuristics, dim=0)

        h_glob_rpt = feature_dict["glob"].repeat_interleave(self.num_resource_heuristics, dim=0)

        action_indices = torch.LongTensor(range(self.num_resource_heuristics))
        resource_heuristic_actions = self.embedding_model(action_indices)
        resource_heuristic_actions = resource_heuristic_actions.repeat_interleave(
            feature_dict['glob'].shape[0], dim=0)

        status_inputs = torch.cat([x_h_dag_rpt, h_glob_rpt, resource_heuristic_actions], dim=1)

        resource_heuristic_scores = self.mlp_score(status_inputs).squeeze(-1)
        return resource_heuristic_scores



