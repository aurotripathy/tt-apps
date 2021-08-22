# hello world inspired from
# https://nestedsoftware.com/2019/08/15/pytorch-hello-world-37mo.156165.html
# just one step of the forward pass
# one input, one hidden, and one putput

import torch
import torch.nn as nn
import torch.nn.functional as F

import sage.api as tt_api # Import Tenstorrent Sage compiler's public API

def get_model(
    par_strategy="RowParallel",
    core_count=4,
):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hidden_layer = nn.Linear(1, 1)
            self.hidden_layer.weight = torch.nn.Parameter(torch.tensor([[1.58]],
                                                                       dtype=torch.float32))
            self.hidden_layer.bias = torch.nn.Parameter(torch.tensor([-0.14], dtype=torch.float32))

            self.output_layer = nn.Linear(1, 1)
            self.output_layer.weight = torch.nn.Parameter(torch.tensor([[2.45]], dtype=torch.float32))
            self.output_layer.bias = torch.nn.Parameter(torch.tensor([-0.11], dtype=torch.float32))

        def forward(self, x):
            x = F.relu(self.hidden_layer(x))
            x = F.relu(self.output_layer(x))
            print('output', x)
            return x

    model = Net().eval()
    
    # Set parallelization strategy
    tt_api.set_parallelization_strat(
        model, cores=((0, 0), (core_count - 1, 0)), strategy=par_strategy
    )

    return model
