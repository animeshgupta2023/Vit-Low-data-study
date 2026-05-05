import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss that accepts soft targets (e.g., from Mixup). Used with timm's Mixup.
    Equivalent to timm.loss.SoftTargetCrossEntropy but written here for self‑contained use.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        loss = torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1)
        return loss.mean()


def kd_loss_fn(student_logits, teacher_logits, temperature=3.0):
    """
    Standard KL‑divergence distillation loss.
    Args:
        student_logits: logits from the student model
        teacher_logits: logits from the frozen teacher model
        temperature: softening temperature
    Returns:
        scalar loss
    """
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    loss = kl_loss(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    return loss


def feature_alignment_loss(student_features, teacher_features):
    """Simple MSE loss between student and teacher feature vectors (e.g., CLS token)."""
    return F.mse_loss(student_features, teacher_features)