import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, input_dim, output_dim):

        self.activation = nn.SiLU()
        self.input_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.activation(x)
        return x
      
        


class Model(nn.Module):
    def __init__(self, input_dim=128, action_space=90, dim=512, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, dim)

        self.layers = (Layer(dim, dim), for _ in range(num_layers))
      
        self.output_proj = nn.Linear(dim, action_space)


    def forward(self, x):
      
        x = self.input_proj(x)

        x = self.layers(x)

        x = self.output_proj(x)

        return x
        
