import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, h_i, h_j, weight=None):
        N =self.batch_size
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)

        nominator = torch.exp(positives)
        denominator = (mask.bool()) * torch.exp(similarity_matrix)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / N
        loss = weight * loss if weight is not None else loss

        return loss
    
class LableLoss(nn.Module):
    def __init__(self, batch_size,class_num, temperature, device):
        super(LableLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.class_num = class_num

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask
       
    def forward_label(self, q_i, q_j,weight=None):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        loss = weight * loss if weight is not None else loss
        return loss + entropy

