import torch

class RolloutBuffer():
    def __init__(self):


        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.rollout_len, self.n_agents, self.dim), dtype=torch.float16, device=self.device)
        self.action = torch.zeros((self.rollout_len, self.n_agents), dtype=torch.long, device=self.device)
        self.log_prob = torch.zeros((self.rollout_len, self.n_agents), dtype=torch.float16, device=self.device)

        self.value = torch.zeros((self.rollout_len, self.n_agents), dtype=torch.float32, device=self.device)
        self.reward = torch.zeros((self.rollout_len, self.n_agents), dtype=torch.float32, device=self.device)
        self.done = torch.zeros((self.rollout_len, self.n_agents), dtype=torch.float16, device=self.device)

        self.returns = torch.zeros((self.rollout_len, self.n_agents), dtype=torch.float32, device=self.device)
        self.advantage = torch.zeros((self.rollout_len, self.n_agents), dtype=torch.float32, device=self.device)

        self.ptr = 0
      
    def store(self, obs, action, log_prob, value, reward, done, returns):



    @torch.no_grad()
    def compute_gae(self, last_value, gae_lambda, gae_gamma):
        last_value = last_value.view(self.n_agents)

        gae = torch.zeros(self.n_agents, device=self.device)

        for t reversed(range(self.rollout_len)):
            non_terminal = 1.0 - self.done[t]
            next_val = last_val 
  
        
