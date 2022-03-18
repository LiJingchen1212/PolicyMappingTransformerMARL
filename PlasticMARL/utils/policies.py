import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import onehot_from_logits, categorical_sample
from itertools import chain
import numpy as np
from torch.distributions import Normal
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
class BasePolicy(nn.Module):
    """
    Base policy network
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations (optionally a tuple that
                                additionally includes a onehot label)
        Outputs:
            out (PyTorch Matrix): Actions
        """
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out


class DiscretePolicy(BasePolicy):
    """
    Policy Network for discrete action spaces
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False,
                return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)
        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets

class PlasticPolicy(nn.Module):
    def __init__(self, num_in_pol, num_out_pol, nagents,
                                     hidden_dim, attend_heads,norm_in=True):
        super(PlasticPolicy, self).__init__()
        self.action_scale = 1
        self.action_bias = 0
        assert (hidden_dim % attend_heads) == 0
        self.nagents = nagents
        self.attend_heads = attend_heads
        self.actor = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, nagents)
        self.log_std_linear = nn.Linear(hidden_dim, nagents)
        self.obs_encoders = nn.ModuleList()

        for i in range(num_in_pol//nagents):
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(nagents,
                                                            affine=False))
            encoder.add_module('enc_fc', nn.Linear(nagents, hidden_dim))
            encoder.add_module('enc_n', nn.LeakyReLU())
            self.obs_encoders.append(encoder)

        self.pre_act_encoder = nn.Sequential()
        self.pre_act_encoder.add_module('pre_act_fc', nn.Linear(hidden_dim,hidden_dim))
        self.pre_act_encoder.add_module('pre_act_n', nn.LeakyReLU())

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()

        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                 attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.obs_encoders]
        """
        self.actor = nn.Sequential()
        self.actor.add_module('act_fc1', nn.Linear(hidden_dim,
                                                  hidden_dim))
        self.actor.add_module('act_nl', nn.LeakyReLU())
        self.actor.add_module('act_fc2', nn.Linear(hidden_dim, nagents))
        """
    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps):
        """
        inps:[state_set, pre_actions]
        state_set: a list of reshaped observation
        pre_actions: the previous actions (Tensor)
        """

        states_set, pre_actions = inps
        #print(states_set)
        states = [s for s in states_set] # 18,8,3

        s_encodings = [encoder(s) for encoder, s in zip(self.obs_encoders, states)] # 18, 8,hidden

        all_head_keys = [[k_ext(enc) for enc in s_encodings] for k_ext in self.key_extractors]

        all_head_values = [[v_ext(enc) for enc in s_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        #print(pre_actions)
        pre_actions = self.pre_act_encoder(pre_actions)
        all_head_selectors = [sel_ext(pre_actions)
                              for sel_ext in self.selector_extractors]

        all_values = []
        all_attend_logits = []
        all_attend_probs = []
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):

            keys = [k for k in curr_head_keys]
            values = [v for v in curr_head_values]
            # calculate attention across agents
            #print(curr_head_selectors.shape,torch.stack(keys).permute(1, 2, 0).shape)

            attend_logits = torch.matmul(curr_head_selectors.view(curr_head_selectors.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
            #print(attend_logits.shape)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            attention_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
            all_values.append(attention_values)
            all_attend_logits.append(attend_logits)
            all_attend_probs.append(attend_weights)

        #head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                           #.mean()) for probs in all_attend_probs]

        actor_in = torch.cat(all_values, dim=1)
        x = F.relu(self.actor(actor_in))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, actor_in

    def sample(self, inps, test=False):
        mean, log_std, actor_in = self.forward(inps)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, actor_in
