import random, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.utils import clamp_probs
from torch_scatter import segment_csr
from gymnasium.core import ObsType, ActType
import torch_geometric.utils as pyg_utils
import numpy as np

from ..scheduler import Scheduler
from spark_sched_sim import graph_utils
from ..heuristic.heuristic import HeuristicScheduler
from ..heuristic.random_scheduler import RandomScheduler
from ..heuristic.fifo import FifoScheduler
from ..heuristic.wscpt import WscptScheduler
from ..heuristic.mc import McScheduler
from ..heuristic.sjf import SjfScheduler
from ..heuristic.ljf import LjfScheduler
from ..heuristic.resource_heuristics import ResourceHeuristics


class NeuralScheduler(Scheduler):
    """Base class for all neural schedulers"""

    def __init__(
        self,
        name,
        actor,
        obs_wrapper_cls,
        num_executors,
        state_dict_path,
        opt_cls,
        opt_kwargs,
        max_grad_norm,
        num_heuristics= 2,
        list_heuristics = ['example1','example2'],
        num_resource_heuristics= 2,
        list_resource_heuristics= ['example1', 'example2'],
        resource_allocation = 'Random'
        ):
        super().__init__(name)

        self.actor = actor
        self.obs_wrapper_cls = obs_wrapper_cls
        self.num_executors = num_executors
        self.opt_cls = opt_cls
        self.opt_kwargs = opt_kwargs
        self.max_grad_norm = max_grad_norm
        self.num_heuristics = num_heuristics
        self.list_heuristics = list_heuristics
        self.resource_allocation = resource_allocation
        self.num_resource_heuristics = num_resource_heuristics
        self.list_resource_heuristics = list_resource_heuristics

        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            self.actor.load_state_dict(state_dict)

        if self.name == "HyperHeuristic":
            self.heuristics_count = [0 for i in range(self.num_heuristics)]
            self.same_action_count = 0
            self.different_action_count = 0

        if self.resource_allocation == "HyperHeuristic":
            self.resource_heuristics_count = [0 for i in range(self.num_resource_heuristics)]

        self.np_random = np.random.RandomState(42)

    @property
    def device(self):
        return next(self.actor.parameters()).device

    def train(self):
        """call only on an instance that is about to be trained"""
        assert self.opt_cls, "optimizer was not specified."
        self.actor.train()
        opt_cls = getattr(torch.optim, self.opt_cls)
        opt_kwargs = self.opt_kwargs or {}
        self.optim = opt_cls(self.actor.parameters(), **opt_kwargs)

    @torch.no_grad()
    def schedule(self, obs: ObsType) -> ActType:
        dag_batch = graph_utils.obs_to_pyg(obs)
        stage_to_job_map = dag_batch.batch
        stage_mask = dag_batch["stage_mask"]

        dag_batch.to(self.device, non_blocking=True)

        # 1. compute node, dag, and global representations
        h_dict = self.actor.encoder(dag_batch)
        # 2. select a schedulable stage
        if self.name == "HyperHeuristic":
            # 2. select a heuristic & retrieve information of the stage selected by the heuristic
            heuristic_score = self.actor.heuristic_policy_network(dag_batch,h_dict)
            heuristic_idx, lgprob = self._sample(heuristic_score)
            if heuristic_idx == 0:
                scheduler = McScheduler(self.num_executors, self.resource_allocation)
            elif heuristic_idx == 1:
                scheduler = WscptScheduler(self.num_executors, self.resource_allocation)
            elif heuristic_idx == 2:
                scheduler = SjfScheduler(self.num_executors, self.resource_allocation)
            elif heuristic_idx == 3:
                scheduler = LjfScheduler(self.num_executors,self.resource_allocation)
            elif heuristic_idx == 4:
                scheduler = FifoScheduler(self.num_executors, self.resource_allocation)
            else:
                sys.exit("Heuristic idx is not matched to any scheduler")
            self.heuristics_count[heuristic_idx] += 1
            action = scheduler(obs)
            stage_idx = action['stage_idx']

            # Compare actions from MCScheduler and WSCPTScheduler
            mc_scheduler = McScheduler(self.num_executors, self.resource_allocation)
            wscpt_scheduler = WscptScheduler(self.num_executors, self.resource_allocation)
            mc_action = mc_scheduler(obs)
            wscpt_action = wscpt_scheduler(obs)

            if mc_action == wscpt_action:
                self.same_action_count += 1
            else:
                self.different_action_count += 1

            #print(f"Same actions: {self.same_action_count}, Different actions: {self.different_action_count}")

        else:
            stage_scores = self.actor.stage_policy_network(dag_batch, h_dict)
            stage_idx, lgprob = self._sample(stage_scores)
            heuristic_idx = -1

        # 3. retrieve index of selected stage's job
        stage_idx_glob = pyg_utils.mask_to_index(stage_mask)[stage_idx]
        job_idx = stage_to_job_map[stage_idx_glob].item()

        # 4. select the number of executors to add to that stage, conditioned
        # on that stage's job & Calculate lgprob
        if self.resource_allocation == "HyperHeuristic":
            resource_heuristic_score = self.actor.resource_heuristic_policy_network(dag_batch, h_dict, job_idx)
            resource_heuristic_idx, resource_lgprob = self._sample(resource_heuristic_score)
            self.resource_heuristics_count[resource_heuristic_idx ] += 1
            num_exec = ResourceHeuristics(resource_heuristic_idx,obs,job_idx)
            lgprob = lgprob + resource_lgprob
        else:
            resource_heuristic_idx = -1
            if self.resource_allocation == 'Random':
                num_exec = self.np_random.randint(0, obs["num_committable_execs"])
            elif self.resource_allocation == 'DNN':
                exec_scores = self.actor.exec_policy_network(dag_batch, h_dict, job_idx)
                num_exec, exec_lgprob = self._sample(exec_scores)
                lgprob = lgprob + exec_lgprob
            elif self.resource_allocation == 'DRA':
                num_exec = action['num_exec']
            else:
                sys.exit("Check -resource allocation parameter.")

        action = {
            'heuristic_idx': heuristic_idx,
            'resource_heuristic_idx': resource_heuristic_idx,
            'stage_idx': stage_idx,
            'job_idx': job_idx,
            'num_exec': num_exec
        }

        return action, lgprob

    def softmax_with_temperature(self, logits, temperature=0.5):
        scaled_logits = logits / temperature
        max_logits = torch.max(scaled_logits)  # For numerical stability
        exp_logits = torch.exp(scaled_logits - max_logits)
        return exp_logits / torch.sum(exp_logits)

    def _sample(self,logits):
        #pi = self.softmax_with_temperature(logits,temperature=0.1).detach().numpy()
        pi = F.softmax(logits, 0).numpy()
        try:
            pi_percentages = [int(p * 100) for p in pi]
        except ValueError:
                print("Error: Unable to calculate action probabilities. logits:",logits)

        #print("action probability:", pi_percentages)
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob

    def evaluate_actions(self, dag_batch, actions):
        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        heuristic_selections, resource_heuristic_selections, stage_selections, job_indices, exec_selections = \
            [col.clone() for col in actions.T]

        num_stage_acts = dag_batch["num_stage_acts"]
        num_exec_acts = dag_batch["num_exec_acts"]
        num_nodes_per_obs = dag_batch["num_nodes_per_obs"]
        obs_ptr = dag_batch["obs_ptr"]
        job_indices += obs_ptr[:-1]

        # re-feed all the observations into the model with grads enabled
        dag_batch.to(self.device)
        h_dict = self.actor.encoder(dag_batch)

        action_lgprobs = 0
        action_entropies = 0

        #evaluate priority rule model
        if self.name == "Decima":
            stage_scores = self.actor.stage_policy_network(dag_batch, h_dict)
            stage_lgprobs, stage_entropies = self._evaluate(
                stage_scores.cpu(), num_stage_acts, stage_selections)
            action_lgprobs += stage_lgprobs
            action_entropies += stage_entropies
        elif self.name == "HyperHeuristic":
            heuristic_score = self.actor.heuristic_policy_network(dag_batch,h_dict)
            heuristic_lgprobs, heuristic_entropies = self._evaluate(
                heuristic_score.cpu(), torch.tensor([self.num_heuristics] * len(job_indices)), heuristic_selections)
            action_lgprobs += heuristic_lgprobs
            action_entropies += heuristic_entropies


        #evaluate resource allocation model
        if self.resource_allocation == "DNN":
            exec_scores = self.actor.exec_policy_network(dag_batch, h_dict, job_indices)
            exec_lgprobs, exec_entropies = self._evaluate(
                exec_scores.cpu(), num_exec_acts[job_indices], exec_selections)
            action_lgprobs += exec_lgprobs
            action_entropies += exec_entropies
        elif self.resource_allocation == "HyperHeuristic":
            resource_heuristic_scores = self.actor.resource_heuristic_policy_network(dag_batch, h_dict, job_indices)
            resource_heuristic_lgprobs, resource_heuristic_entropies = self._evaluate(
                resource_heuristic_scores.cpu(), torch.tensor([self.num_resource_heuristics] * len(job_indices)), resource_heuristic_selections)
            action_lgprobs += resource_heuristic_lgprobs
            action_entropies += resource_heuristic_entropies

        # Normalize entropies
        if self.name == "HyperHeuristic":
            action_entropies /= (self.num_executors * torch.tensor([self.num_heuristics * len(job_indices)])).log()
        else:
            action_entropies /= (self.num_executors * num_nodes_per_obs).log()

        return action_lgprobs, action_entropies

    @classmethod
    def _evaluate(cls, scores, counts, selections):
        ptr = counts.cumsum(0)
        ptr = torch.cat([torch.tensor([0]), ptr], 0)
        selections += ptr[:-1]
        probs = pyg_utils.softmax(scores, ptr=ptr)
        probs = clamp_probs(probs)
        log_probs = probs.log()
        selection_log_probs = log_probs[selections]
        entropies = -segment_csr(log_probs * probs, ptr)
        return selection_log_probs, entropies

    def update_parameters(self, loss=None):
        # initial_embeddings = self.actor.embedding_model.embedding.weight.data.clone()
        # initial_mlp_score_weights = self.actor.heuristic_policy_network.mlp_score[0].weight.data.clone()

        if loss:
            # accumulate gradients
            loss.backward()

        if self.max_grad_norm:
            # clip grads
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm, error_if_nonfinite=True
                )
            except RuntimeError:
                print("infinite grad; skipping update.")
                return
        # check the gradient
        # for name, param in self.actor.named_parameters():
        #     if param.requires_grad:
        #         if param.grad is not None:
        #             print(f"Gradient for {name}: {param.grad.mean().item()}")
        #         else:
        #             print(f"No gradient computed for {name}")
        # update model parameters
        self.optim.step()

        # clear accumulated gradients
        self.optim.zero_grad()

        # updated_embeddings = self.actor.embedding_model.embedding.weight.data
        # updated_mlp_score_weights = self.actor.heuristic_policy_network.mlp_score[0].weight.data
        # print(f"Initial embeddings: {initial_embeddings}")
        # print(f"Updated embeddings: {updated_embeddings}")
        #
        # Check differences
        # diff_embeddings = updated_embeddings - initial_embeddings
        # diff_mlp_score_weights = updated_mlp_score_weights - initial_mlp_score_weights
        #print(f"Difference in embeddings: {diff_embeddings}")
        #print(f"Difference in mlp_score weights: {diff_mlp_score_weights}")

def make_mlp(input_dim, hid_dims, output_dim, act_cls, act_kwargs=None):
    if isinstance(act_cls, str):
        act_cls = getattr(torch.nn.modules.activation, act_cls)

    mlp = nn.Sequential()
    prev_dim = input_dim
    hid_dims = hid_dims + [output_dim]
    for i, dim in enumerate(hid_dims):
        mlp.append(nn.Linear(prev_dim, dim))
        if i == len(hid_dims) - 1:
            break
        act_fn = act_cls(**act_kwargs) if act_kwargs else act_cls()
        mlp.append(act_fn)
        prev_dim = dim
    return mlp


class StagePolicyNetwork(nn.Module):
    def __init__(self, num_node_features, emb_dims, mlp_kwargs):
        super().__init__()
        input_dim = (
            num_node_features + emb_dims["node"] + emb_dims["dag"] + emb_dims["glob"]
        )

        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, dag_batch, h_dict):
        stage_mask = dag_batch['stage_mask']

        x = dag_batch.x[stage_mask]

        h_node = h_dict['node'][stage_mask]

        batch_masked = dag_batch.batch[stage_mask]
        h_dag_rpt = h_dict['dag'][batch_masked]

        try:
            num_stage_acts = dag_batch['num_stage_acts'] # batch of obsns
        except:
            num_stage_acts = stage_mask.sum() # single obs

        h_glob_rpt = h_dict['glob'].repeat_interleave(
            num_stage_acts, output_size=h_node.shape[0], dim=0)

        # residual connections to original features
        node_inputs = torch.cat([x, h_node, h_dag_rpt, h_glob_rpt], dim=1)

        node_scores = self.mlp_score(node_inputs).squeeze(-1)
        return node_scores


class ExecPolicyNetwork(nn.Module):
    def __init__(self, num_executors, num_dag_features, emb_dims, mlp_kwargs):
        super().__init__()
        self.num_executors = num_executors
        self.num_dag_features = num_dag_features
        input_dim = num_dag_features + emb_dims["dag"] + emb_dims["glob"] + 1

        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, dag_batch, h_dict, job_indices):
        exec_mask = dag_batch["exec_mask"]

        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, : self.num_dag_features]
        x_dag = x_dag[job_indices]

        h_dag = h_dict["dag"][job_indices]

        exec_mask = exec_mask[job_indices]

        try:
            # batch of obsns
            num_exec_acts = dag_batch["num_exec_acts"][job_indices]
        except KeyError:
            # single obs
            num_exec_acts = exec_mask.sum()
            x_dag = x_dag.unsqueeze(0)
            h_dag = h_dag.unsqueeze(0)
            exec_mask = exec_mask.unsqueeze(0)

        exec_actions = self._get_exec_actions(exec_mask)

        # residual connections to original features
        x_h_dag = torch.cat([x_dag, h_dag], dim=1)
        x_h_dag_rpt = x_h_dag.repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], dim=0)
        h_glob_rpt = h_dict["glob"].repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], dim=0)
        dag_inputs = torch.cat([x_h_dag_rpt, h_glob_rpt, exec_actions], dim=1)
        dag_scores = self.mlp_score(dag_inputs).squeeze(-1)

        return dag_scores

    def _get_exec_actions(self, exec_mask):
        exec_actions = torch.arange(self.num_executors) / self.num_executors
        exec_actions = exec_actions.to(exec_mask.device)
        exec_actions = exec_actions.repeat(exec_mask.shape[0])
        exec_actions = exec_actions[exec_mask.view(-1)]
        exec_actions = exec_actions.unsqueeze(1)
        return exec_actions

class HeuristicPolicyNetwork(nn.Module):
    def __init__(self, embedding_model, num_heuristics,
        list_heuristics, input_feature, emb_dims, mlp_kwargs):

        super().__init__()
        self.num_heuristics = num_heuristics
        self.list_heuristics = list_heuristics
        self.embedding_model = embedding_model
        self.input_feature = input_feature

        self.total_feature_list = ["num_queue","glob","cpt_mean","cpt_var","children_mean","children_var"]
        self.num_queue_max = 1000
        num_queue_emb_dim = 3
        self.num_queue_embedding = nn.Embedding(self.num_queue_max + 1, num_queue_emb_dim)
        #self.num_queue_embedding.weight.requires_grad = False
        #print(self.num_queue_embedding.weight.requires_grad)

        feature_in_use = [feature in self.input_feature for feature in self.total_feature_list]
        dim_feature_list = [num_queue_emb_dim, emb_dims['glob'], 1,1,1,1]
        input_dim = sum(dim for dim, use in zip(dim_feature_list, feature_in_use) if use) + emb_dims['heuristic']

        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, dag_batch, h_dict):
        input_matrix = []

        if "glob" in self.input_feature:
            h_glob_rpt = h_dict['glob'].repeat_interleave(self.num_heuristics, dim=0)
            input_matrix.append(h_glob_rpt)

        if "num_queue" in self.input_feature:
            num_queue = min(torch.sum(dag_batch["stage_mask"]),torch.tensor(self.num_queue_max)).long()
            num_queue_emb = self.num_queue_embedding(num_queue).repeat(h_glob_rpt.shape[0], 1)
            input_matrix.append(num_queue_emb)

        if "cpt_mean" in self.input_feature or "cpt_var" in self.input_feature:
            stage_mask = dag_batch["stage_mask"].bool()
            stage_cpt = dag_batch.x[:,5]

            masked_stages_cpt = stage_cpt[stage_mask]

            if "cpt_mean" in self.input_feature:
                mean_cpt = torch.mean(masked_stages_cpt).repeat(h_glob_rpt.shape[0],1)
                input_matrix.append(mean_cpt)
            if "cpt_var" in self.input_feature:
                var_cpt = torch.std(masked_stages_cpt).repeat(h_glob_rpt.shape[0],1)
                input_matrix.append(var_cpt)

        if "children_mean" in self.input_feature or "children_var" in self.input_feature:
            stage_mask = dag_batch["stage_mask"].bool()
            stage_children = dag_batch.x[:,6]

            masked_stages_children = stage_children[stage_mask ]

            if "children_mean" in self.input_feature:
                mean_children = torch.mean(masked_stages_children).repeat(h_glob_rpt.shape[0],1)
                input_matrix.append(mean_children)
            if "children_var" in self.input_feature:
                var_children = torch.std(masked_stages_children).repeat(h_glob_rpt.shape[0],1)
                input_matrix.append(var_children)

        #print("input_matrix:",torch.cat(input_matrix, dim=1))
        action_indices = torch.LongTensor(range(self.num_heuristics))
        heuristic_actions = self.embedding_model(action_indices)
        heuristic_actions = heuristic_actions.repeat_interleave(h_dict['glob'].shape[0], output_size=h_glob_rpt.shape[0], dim=0)
        input_matrix.append(heuristic_actions)

        # residual connections to original features
        status_inputs = torch.cat(input_matrix, dim=1)

        heuristic_scores = self.mlp_score(status_inputs).squeeze(-1)
        #print("heuristic_actions",heuristic_actions)
        #print("num_queue:",num_queue[0][0].item(),",heuristic_scores:",heuristic_scores)
        return heuristic_scores


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

    def forward(self, dag_batch, h_dict, job_indices):
        dag_start_idxs = dag_batch.ptr[:-1]
        x_dag = dag_batch.x[dag_start_idxs, : self.num_dag_features]
        x_dag = x_dag[job_indices]
        h_dag = h_dict["dag"][job_indices]

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

        h_glob_rpt = h_dict["glob"].repeat_interleave(self.num_resource_heuristics, dim=0)

        action_indices = torch.LongTensor(range(self.num_resource_heuristics))
        resource_heuristic_actions = self.embedding_model(action_indices)
        resource_heuristic_actions = resource_heuristic_actions.repeat_interleave(
            h_dict['glob'].shape[0], dim=0)

        status_inputs = torch.cat([x_h_dag_rpt, h_glob_rpt, resource_heuristic_actions], dim=1)

        resource_heuristic_scores = self.mlp_score(status_inputs).squeeze(-1)
        return resource_heuristic_scores



