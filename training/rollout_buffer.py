import torch
from config import RLConfig

class RolloutBuffer():
    def __init__(self, config: RLConfig):
        self.rollout_len = config.rollout_len

        self.dim = config.n_envs * config.n_agents

        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.rollout_len, self.dim, self.obs_dim), dtype=torch.float16, device=self.device)
        self.action = torch.zeros((self.rollout_len, self.dim), dtype=torch.long, device=self.device)
        self.log_prob = torch.zeros((self.rollout_len, self.dim), dtype=torch.float16, device=self.device)

        self.value = torch.zeros((self.rollout_len, self.dim), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((self.rollout_len, self.dim), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((self.rollout_len, self.dim), dtype=torch.float16, device=self.device)

        self.returns = torch.zeros((self.rollout_len, self.dim), dtype=torch.float32, device=self.device)
        self.advantage = torch.zeros((self.rollout_len, self.dim), dtype=torch.float32, device=self.device)

        self.ptr = 0
      
    def store(self, obs, action, log_prob, value, reward, done, returns):
        if self.ptr >= self.rollout_len: return

        idx = self.ptr

        self.obs[idx].copy_(obs.view(self.dim, -1))
        self.action[idx].copy_(action.view(self.dim, -1))
        self.log_prob[idx].copy_(log_prob.view(self.dim, -1))
        self.value[idx].copy_(value.view(self.dim, -1))
        self.reward[idx].copy_(reward.view(self.dim, -1))
        self.done[idx].copy_(done.view(self.dim, -1))

        self.ptr += 1

    @torch.no_grad()
    def compute_gae(self, last_value, gae_lambda, gae_gamma):
        last_value = last_value.view(self.dim)

        gae = torch.zeros(self.dim, device=self.device)

        for t reversed(range(self.rollout_len)):
            non_terminal = 1.0 - self.done[t]
            next_val = last_val if t == self.rollout_len - 1 else self.value[t + 1]

            delta = self.reward[t] + gamma_gae * next_val * non_terminal - self.value[t]
            gae = delta + gae_gamma * gae_lambda * non_terminal * gae

            self.advantages[t] = gae
            self.returns[t] = gae + self.value[t]
        
        
