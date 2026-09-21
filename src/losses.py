import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, alpha=None,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)

        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        focal_weight = (1.0 - probs) ** self.gamma

        loss = -focal_weight * smooth_targets * log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha.unsqueeze(0).expand_as(loss)
            loss = alpha_weight * loss

        loss = loss.sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(config, class_weights=None):
    if config.USE_FOCAL_LOSS:
        return FocalLoss(
            gamma=config.FOCAL_LOSS_GAMMA,
            alpha=class_weights if class_weights is not None else config.FOCAL_LOSS_ALPHA,
            label_smoothing=config.LABEL_SMOOTHING,
        )
    else:
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING,
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))

    ce = nn.CrossEntropyLoss()
    fl = FocalLoss(gamma=2.0)

    print(f"CrossEntropy Loss: {ce(logits, targets):.4f}")
    print(f"Focal Loss (γ=2):  {fl(logits, targets):.4f}")

    fl_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
    print(f"Focal + Smooth:    {fl_smooth(logits, targets):.4f}")
    print("All tests passed!")
