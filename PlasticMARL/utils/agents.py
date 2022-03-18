from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies import DiscretePolicy, PlasticPolicy

class PlasticAgent(object):

    def __init__(self, nagents,num_in_pol, num_out_pol,action_space, hidden_dim=64,
                 lr=0.01, attend_heads=4, onehot_dim=0):

        self.policy = PlasticPolicy(num_in_pol=num_in_pol, num_out_pol=num_out_pol, nagents=nagents,
                                     hidden_dim=hidden_dim,
                                     attend_heads = attend_heads)
        self.target_policy = PlasticPolicy(num_in_pol=num_in_pol, num_out_pol=num_out_pol, nagents=nagents,
                                     hidden_dim=hidden_dim,
                                     attend_heads = attend_heads)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):

        if explore:
            return self.policy.sample(obs)
        else:
            return self.policy(obs)


    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
