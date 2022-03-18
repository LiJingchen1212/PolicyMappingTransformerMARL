import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import PlasticAgent
from utils.critics import PlasticCritic
from utils.policies import PlasticPolicy

MSELoss = torch.nn.MSELoss()

class PlasticSAC(object):
    def __init__(self, agent_init_params, sa_size, nagents,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 **kwargs):

        self.nagents = nagents
        self.alpha =0.2
        self.agents = [PlasticAgent(lr=pi_lr, nagents=nagents,
                                      hidden_dim=pol_hidden_dim, attend_heads=attend_heads,
                                      **params)
                         for params in agent_init_params]
        self.critic = PlasticCritic(nagents=nagents, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads, **agent_init_params[0])
        self.target_critic = PlasticCritic(nagents=nagents, hidden_dim=critic_hidden_dim,
                                      attend_heads=attend_heads, **agent_init_params[0])
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):

        states,previous = observations
        return [a.step((states,pre), explore=explore) for a,pre in zip(self.agents,previous)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):

        obs, pre_acts, acs, rews, next_obs, next_acts, dones = sample

        next_acs = []
        next_log_pis = []
        for pi, ob, next_act in zip(self.target_policies, next_obs, next_acts):
            curr_next_ac, curr_next_log_pi, _,_ = pi.sample([ob,next_act],test=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = (next_obs, next_acs) #next_acs:2,1024,3
        critic_in = (obs, acs)
        next_qs = self.target_critic(trgt_critic_in,test=True)
        critic_rets = self.critic(critic_in, regularize=True,
                                  logger=logger, niter=self.niter)
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(
            self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs,pre_acts, acs, rews, next_obs,next_acts, dones = sample
        samp_acs = []
        mean_acs = []

        all_log_pis = []


        for a_i, pi, ob, pre_act in zip(range(self.nagents), self.policies, obs,pre_acts):
            curr_ac, log_pi,mean,_= pi.sample([ob,pre_act])

            mean_acs.append(mean)
            samp_acs.append(curr_ac)

            all_log_pis.append(log_pi)


        critic_in = (obs, samp_acs)
        critic_rets = self.critic(critic_in)
        for a_i, log_pi, q in zip(range(self.nagents),all_log_pis, critic_rets):
            curr_agent = self.agents[a_i]
            pol_loss = ((self.alpha * log_pi) - q).mean()
            disable_gradients(self.critic)
            if a_i < self.nagents - 1:
                pol_loss.backward(retain_graph=True)
            else:
                pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(
                curr_agent.policy.parameters(), 0.5)

            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i,
                                  pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i,
                                  grad_norm, self.niter)


    def update_all_targets(self):

        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):

        self.prep_training(device='cpu')
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                      **kwargs):
        agent_init_params = []
        sa_size = []

        nagents = len(env.action_space)
        for i in range(env.action_space[0].shape[0]):

            agent_init_params.append({'num_in_pol': len(env.action_space)*env.observation_space[0].shape[0],
                                      'num_out_pol': len(env.action_space),
                                      'action_space': env.action_space[0].shape[0]}
                                     )

        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size,
                     'nagents': nagents}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):

        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance