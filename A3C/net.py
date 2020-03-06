import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class A3C(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(A3C, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 32)
        self.a2 = nn.Linear(32, a_dim)
        self.v1 = nn.Linear(s_dim, 16)
        self.v2 = nn.Linear(16, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dist = torch.distributions.Categorical

    def forward(self, x):
        a = self.a1(x)
        a = self.relu(a)
        logits = self.a2(a)
        
        v = self.v1(x)
        v = self.relu(v)
        values = self.v2(v)
        
        return logits, values

    def choose_action(self, s):
        self.eval()
        
        logits, _ = self.forward(s)
        probs = F.softmax(logits, dim=1).data
        m = self.dist(probs)
        a = m.sample().numpy()[0]
        
        return a
        
    def loss(self, s, a, v_t):
        self.train()
        
        logits, values = self.forward(s)
        
        td_error = v_t - values
        c_loss = td_error.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.dist(probs)
        a_loss = - m.log_prob(a) * td_error.detach().squeeze()
        
        total_loss = (c_loss + a_loss).mean()
        
        return total_loss
