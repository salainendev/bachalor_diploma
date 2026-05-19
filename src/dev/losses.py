import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- multi head loss -----------
def oracle_loss(preds, targets, reduction='mean'):
    """
    preds: [batch, K, output_dim]
    targets: [batch, output_dim]
    """
    loss_per_head = F.mse_loss(preds, targets.unsqueeze(1).expand_as(preds), reduction='none')
    loss_per_head = loss_per_head.mean(dim=2)       
    min_loss = loss_per_head.min(dim=1)[0]
    return min_loss.mean() if reduction == 'mean' else min_loss.sum()

def diversity_loss(preds):
    head_means = preds.mean(dim=0) 
    sim = F.cosine_similarity(head_means.unsqueeze(0), head_means.unsqueeze(1), dim=2)
    off_diag = sim[~torch.eye(preds.size(1), dtype=bool).to(sim.device)]
    return off_diag.mean()

def confidence_loss(confs, preds, targets):
    loss_per_head = F.mse_loss(preds, targets.unsqueeze(1).expand_as(preds), reduction='none')
    loss_per_head = loss_per_head.mean(dim=2)        
    best_head = torch.argmin(loss_per_head, dim=1)   
    target_conf = F.one_hot(best_head, num_classes=confs.size(1)).float()  
    
    if confs.dim() > 2:
        target_conf = target_conf.unsqueeze(-1).expand_as(confs)  
        
    return F.binary_cross_entropy_with_logits(confs, target_conf)

def confidence_loss_soft(confs, preds, targets, temperature=1.0):
    
    mse_per_head = F.mse_loss(preds, targets.unsqueeze(1).expand_as(preds), reduction='none')
    mse_per_head = mse_per_head.mean(dim=2)  
    soft_labels = torch.exp(-mse_per_head / temperature)
    soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
    return - (soft_labels * F.log_softmax(confs, dim=1)).sum(dim=1).mean()

def compute_total_loss(model, x_batch, y_batch, lambda_div=0.2, lambda_conf=0):
    preds, confs = model(x_batch, return_all=True)
    loss_ora = oracle_loss(preds, y_batch)
    loss_div = diversity_loss(preds)
    loss_conf = confidence_loss(confs, preds, y_batch)
    total = loss_ora + lambda_div * loss_div + lambda_conf * loss_conf
    return total, loss_ora, loss_div, loss_conf

# ------- metric learning loss ---------

class SpectrumParamContrastiveLoss(nn.Module):
    """
    Явный контрастивный лосс для структуры (anchor, pos, neg) без лейблов.
    Притягивает pos к anchor, отталкивает neg от anchor до порога margin.
    """
    def __init__(self, margin=0.2, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, z_anchor: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor):
        # 1. Фиксируем нормы = 1 для стабильного косинуса
        z_anchor = F.normalize(z_anchor, p=2, dim=1)
        z_pos    = F.normalize(z_pos,    p=2, dim=1)
        z_neg    = F.normalize(z_neg,    p=2, dim=1)

        # 2. Косинусные сходства
        cos_pos = F.cosine_similarity(z_anchor, z_pos, dim=1)
        cos_neg = F.cosine_similarity(z_anchor, z_neg, dim=1)

        # 3. Разделённые потери (нет конкуренции градиентов)
        loss_pos = (1.0 - cos_pos)                      # тянем к 1.0
        loss_neg = F.relu(cos_neg - self.margin)        # толкаем ниже margin

        loss = self.pos_weight * loss_pos + self.neg_weight * loss_neg
        return loss.mean()