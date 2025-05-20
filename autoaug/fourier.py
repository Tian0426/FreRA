import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class FreRA(Module):
    def __init__(self, len_sw, device=None, dtype=None) -> None:
        super(FreRA,self).__init__()
        print('Initializing FreRA')
        factory_kwargs = {'device': device, 'dtype': None}

        n_fourier_comp = len_sw //2 + 1
        self.weight = Parameter(torch.empty((n_fourier_comp, 2), **factory_kwargs))
        self.reset_parameters()

    def get_sampling(self, weight, temperature=0.1, bias=0.0):

        if self.training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.cuda()
            gate_inputs = (gate_inputs + weight) / temperature # todo adaptive temperature
            para = torch.sigmoid(gate_inputs)
            return para
        else:
            return torch.sigmoid(weight)


    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.10)
    def forward(self,x, temperature):
        para = self.get_sampling(self.weight, temperature=temperature)
        self.para = para

        noise_para = self.weight.detach().clone() * (-1)
        noise_para[noise_para < max(0, noise_para[:, 0].mean())] = 0.0
        scaling_factor = 1.0 / noise_para[:, 0][noise_para[:, 0] != 0].mean()

        x_ft = torch.fft.rfft(x, dim=-2)
        x_ft = x_ft * torch.unsqueeze(para[:, 0] + noise_para[:, 0]*scaling_factor, -1)
        aug = torch.fft.irfft(x_ft, n=x.shape[-2], dim=-2)

        return aug