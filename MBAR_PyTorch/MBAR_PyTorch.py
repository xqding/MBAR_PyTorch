__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2018/05/29 20:38:05"

import numpy as np
import torch
import torch.nn as nn
import scipy.optimize as optimize

class MBAR():
    def __init__(self, energy, num_conf, cuda = False):
        self.cuda = cuda
        
        self.energy = energy
        self.num_conf = num_conf
        
        self.num_states = energy.shape[0]
        self.tot_num_conf = energy.shape[1]
        
        assert(np.sum(self.num_conf) == self.tot_num_conf)
        assert(self.num_states == len(self.num_conf))

        self.flag_zero = num_conf == 0
        self.flag_nz = num_conf != 0

        self.energy_zero = torch.from_numpy(energy[self.flag_zero, :])
        self.energy_nz = torch.from_numpy(energy[self.flag_nz, :])
        self.num_conf_nz = torch.from_numpy(num_conf[self.flag_nz])
        self.num_states_nz = self.energy_nz.shape[0]

        if self.cuda:
            self.energy_zero = self.energy_zero.cuda()
            self.energy_nz = self.energy_nz.cuda()
            self.num_conf_nz = self.num_conf_nz.cuda()
        
        self.bias_energy_nz = None

        
    def loss_nz(self, bias_energy_nz):
        assert(self.num_states_nz == len(bias_energy_nz))
        bias_energy_nz = torch.tensor(bias_energy_nz,
                                      requires_grad = True,
                                      dtype = self.energy_nz.dtype)
        if self.cuda:
            self.bias_energy_nz = bias_energy_nz.cuda()
        else:
            self.bias_energy_nz = bias_energy_nz
            
        energy_nz = self.energy_nz - torch.min(self.energy_nz, 0)[0]    
        tmp = torch.exp(-(energy_nz +
                          self.bias_energy_nz.view([self.num_states_nz, 1])))        
        tmp = torch.sum(tmp, 0)
        loss = torch.sum(torch.log(tmp)) + torch.sum(self.num_conf_nz*self.bias_energy_nz)

        loss.backward()
        return loss.cpu().detach().numpy().astype(np.float64), bias_energy_nz.cpu().grad.numpy().astype(np.float64)

    def solve(self):
        x0 = self.energy_nz.new(self.num_states_nz).zero_()
        x0 = x0.cpu().numpy()
        
        x, f, d = optimize.fmin_l_bfgs_b(self.loss_nz, x0, iprint = 1)        
        self.bias_energy_nz = self.energy_nz.new(x)
        if self.cuda:
            self.bias_energy_nz = self.bias_energy_nz.cuda()
        
        ## get free energies for states with nonzero number of samples
        sample_prop_nz = self.num_conf_nz / torch.sum(self.num_conf_nz)
        self.F_nz = -torch.log(sample_prop_nz) - self.bias_energy_nz

        ## normalize free energies
        prob_nz = torch.exp(-self.F_nz)
        prob_nz = prob_nz / torch.sum(prob_nz)
        self.F_nz = -torch.log(prob_nz)

        ## update bias energies for states with nonzero number of samples
        ## using normalized free energies
        self.bias_energy_nz = -torch.log(sample_prop_nz) - self.F_nz

        ## calculate free energies for states with zero number of samples
        self.F = self.bias_energy_nz.new(self.num_states)
        
        if self.cuda:
            self.F = self.F.cuda()
            
        idx_zero = 0        
        idx_nz = 0
        for i in range(self.num_states):
            if self.flag_nz[i]:
                self.F[i] = self.F_nz[idx_nz]
                idx_nz += 1
            else:
                tmp = self.energy_nz + self.bias_energy_nz.view((-1,1)) - \
                      self.energy_zero[idx_zero, :]
                tmp = -torch.log(torch.mean(1.0/torch.sum(torch.exp(-tmp), 0)))
                self.F[i] = tmp
                idx_zero += 1
                
        self.F = self.F - self.F[0]
        self.bias_energy_nz = -torch.log(sample_prop_nz) - self.F[torch.ByteTensor(self.flag_nz.astype(int))]
        
        return self.F.cpu().numpy()
