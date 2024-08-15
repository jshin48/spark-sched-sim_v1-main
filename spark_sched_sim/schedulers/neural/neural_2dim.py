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
from ..heuristic.round_robin import RoundRobinScheduler
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
        resource_allocation='Random',
        num_resource_heuristics=2,
        list_resource_heuristics=['example1', 'example2'],
        num_heuristics= 2,
        list_heuristics = ['example1','example2'],
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
            heuristic_score = self.actor.heuristic_policy_network(h_dict)
            heuristic_idx, lgprob = self._sample(heuristic_score)
            if heuristic_idx == 0:
                scheduler = McScheduler(self.num_executors)
            elif heuristic_idx == 1:
                scheduler = WscptScheduler(self.num_executors)
            elif heuristic_idx == 2:
                scheduler = RoundRobinScheduler(self.num_executors, dynamic_partition=False)
            elif heuristic_idx == 3:
                scheduler = SjfScheduler(self.num_executors)
            elif heuristic_idx == 4:
                scheduler = LjfScheduler(self.num_executors)
            else:
                sys.exit("Heuristic idx is not matched to any scheduler")
            self.heuristics_count[heuristic_idx] += 1
            action = scheduler(obs)
            stage_idx = action['stage_idx']
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
            #print("resource_heuristic_idx", resource_heuristic_idx)
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
                num_exec = max(1,min(obs["DRA_exec_cap"][job_idx]-obs["exec_supplies"][job_idx], obs["num_committable_execs"]))-1
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

    def _sample(self, logits):
        pi = F.softmax(logits, 0).numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob

    def evaluate_actions(self, dag_batch, actions):
        # split columns of `actions` into separate tensors
        # NOTE: columns need to be cloned to avoid in-place operation
        #JS Revision
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
            heuristic_score = self.actor.heuristic_policy_network(h_dict)
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

        # update model parameters
        self.optim.step()

        # clear accumulated gradients
        self.optim.zero_grad()


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
            num_exec_acts, output_size=exec_actions.shape[0], dim=0
        )

        h_glob_rpt = h_dict["glob"].repeat_interleave(
            num_exec_acts, output_size=exec_actions.shape[0], dim=0
        )

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
    def __init__(
        self,
        num_heuristics,
        list_heuristics,
        emb_dims,
        mlp_kwargs
    ):
        super().__init__()
        self.num_heuristics = num_heuristics
        self.list_heuristics = list_heuristics
        self.embedding_model = HeuristicEmbeddingModel(num_heuristics, emb_dims['heuristic'])

        # Utilize all available information
        input_dim = emb_dims['glob'] + emb_dims['heuristic']
        self.mlp_score = make_mlp(input_dim, output_dim=1, **mlp_kwargs)

    def forward(self, h_dict):
        h_glob_rpt = h_dict['glob'].repeat_interleave(
            self.num_heuristics, dim=0)
        action_indices = torch.LongTensor(range(self.num_heuristics))
        heuristic_actions = self.embedding_model(action_indices)
        heuristic_actions = heuristic_actions.repeat_interleave(
            h_dict['glob'].shape[0], dim=0)
        # residual connections to original features
        status_inputs = torch.cat(
            [
                h_glob_rpt,
                heuristic_actions
            ],
            dim=1
        )

        heuristic_scores = self.mlp_score(status_inputs).squeeze(-1)

        return heuristic_scores

class HeuristicEmbeddingModel(nn.Module):
    def __init__(self, action_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(action_size, embedding_dim)

    def forward(self, action_indices):
        return self.embedding(action_indices)

class ResourcePolicyNetwork(nn.Module):
    def __init__(self, num_resource_heuristics, list_resource_heuristics,
                 num_executors, num_dag_features, emb_dims, mlp_kwargs):
        super().__init__()
        self.num_executors = num_executors
        self.num_dag_features = num_dag_features
        self.num_resource_heuristics = num_resource_heuristics
        self.list_resource_heuristics = list_resource_heuristics
        self.embedding_model = HeuristicEmbeddingModel(num_resource_heuristics, emb_dims['heuristic'])

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

