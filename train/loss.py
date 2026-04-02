# train/loss.py
"""
Multi-Task Loss Function
Harivignesh — SurgSentinel
Reference implementation — not used in prototype demo
"""
import torch
import torch.nn.functional as F


def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for imbalanced binary classification.
    Handles 8% positive class (high-risk windows) without oversampling.
    Oversampling would corrupt temporal continuity — focal loss is the correct solution.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal = ((1 - pt) ** gamma * bce_loss).mean()
    return focal


def multitask_loss(
    outputs: dict,
    batch: dict,
    lambdas: tuple = (1.0, 0.5, 0.8),
    phase_weights: torch.Tensor = None
) -> dict:
    """
    Joint multi-task loss for SurgSentinel.

    Loss components:
      L_phase:      Cross-entropy over 7 phase classes (primary task, lam=1.0)
      L_instrument: Binary CE over 7 instrument presence labels (regulariser, lam=0.5)
      L_risk:       Focal loss for high-risk window binary prediction (novel, lam=0.8)

    Lambda search grid: {0.5, 1.0, 2.0} x {0.3, 0.5, 1.0} x {0.5, 0.8, 1.2}
    Instrument loss kept lower to prevent auxiliary task dominating during early epochs.
    """
    lam_phase, lam_instr, lam_risk = lambdas

    # Phase loss: weighted CE for class imbalance
    L_phase = F.cross_entropy(
        outputs['phase'].view(-1, 7),
        batch['phases'].view(-1),
        weight=phase_weights
    )

    # Instrument loss: multi-label binary CE
    L_instr = F.binary_cross_entropy(
        torch.sigmoid(outputs['instrument']),
        batch['instruments'].float()
    )

    # Risk loss: focal loss for high-risk window detection
    L_risk = focal_loss(
        outputs['risk'],
        batch['risk'].float(),
        gamma=2.0
    )

    L_total = lam_phase * L_phase + lam_instr * L_instr + lam_risk * L_risk

    return {
        'total': L_total,
        'phase': L_phase.item(),
        'instrument': L_instr.item(),
        'risk': L_risk.item(),
    }
