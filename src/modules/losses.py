"""
Loss Functions - Exact from Notebook
From: final-application-maxwell-for-segmentation-task (3).ipynb
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss.
    Combines Tversky Index to handle class imbalance and Focal Loss to focus on hard samples.
    """
    def __init__(self, 
                 num_classes: int, 
                 alpha: float = 0.3, 
                 beta: float = 0.7, 
                 gamma: float = 4.0 / 3.0, 
                 epsilon: float = 1e-6):
        """
        Args:
            num_classes (int): Number of segmentation classes (including background).
            alpha (float): Weight for False Positives (FP).
            beta (float): Weight for False Negatives (FN).
            gamma (float): Focal parameter. Value > 1 to focus on hard samples.
            epsilon (float): Small constant to avoid division by zero.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Đầu ra raw từ model, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth, shape (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        class_losses = []
        for class_idx in range(1, self.num_classes):
            pred_class = probs[:, class_idx, :, :]
            target_class = targets_one_hot[:, class_idx, :, :]
            
            # Flatten tensor for calculation
            pred_flat = pred_class.contiguous().view(-1)
            target_flat = target_class.contiguous().view(-1)

            # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
            tp = torch.sum(pred_flat * target_flat)
            fp = torch.sum(pred_flat * (1 - target_flat))
            fn = torch.sum((1 - pred_flat) * target_flat)
            
            # Calculate Tversky Index (TI)
            tversky_index = (tp + self.epsilon) / (tp + self.alpha * fp + self.beta * fn + self.epsilon)
            
            # Calculate Focal Tversky Loss (FTL) for current class
            # **Use modified and verified formula: (1 - TI)^γ**
            focal_tversky_loss = torch.pow(1 - tversky_index, self.gamma)
            
            class_losses.append(focal_tversky_loss)
            
        # Average loss of foreground classes
        if not class_losses:
             return torch.tensor(0.0, device=logits.device) # Avoid error if only 1 class

        total_loss = torch.mean(torch.stack(class_losses))
        
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    Inherited from https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py
    """
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        Args:
            gamma (float): Focal parameter. Higher value focuses more on hard samples.
            alpha (torch.Tensor, optional): Weight for each class, shape (C,).
            reduction (str, optional): 'mean', 'sum' or 'none'.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Đầu ra raw từ model, shape (B, C, H, W).
            targets (torch.Tensor): Ground truth, shape (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Calculate original CE loss
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        
        # Get probability of correct class (p_t)
        # pt.shape: (B, H, W)
        pt = torch.exp(-ce_loss)
        
        # Calculate Focal Loss
        # (1-pt)^gamma * ce_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            
            # Get alpha corresponding to each pixel
            alpha_t = self.alpha.gather(0, targets.view(-1)).view_as(targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss based on Helmholtz equation residual.
    """
    def __init__(self, scale_factor=0.01):
        super().__init__()
        omega, mu_0, eps_0 = 2 * np.pi * 42.58e6, 4 * np.pi * 1e-7, 8.854187817e-12
        self.k0 = torch.tensor(omega * np.sqrt(mu_0 * eps_0), dtype=torch.float32)
        self.scale_factor = scale_factor
        
    def forward(self, b1, eps, sig):
        from src.utils.helpers import compute_helmholtz_residual
        
        eps = eps.to(b1.device)
        sig = sig.to(b1.device)
        
        residual = compute_helmholtz_residual(b1, eps, sig, self.k0)
        return torch.mean(residual) * self.scale_factor

class DynamicLossWeighter(nn.Module):
    """
    Automatically adjusts weights for multiple loss components,
    ensuring sum of weights always equals 1 using Softmax.
    """
    def __init__(self, num_losses: int, tau: float = 1.0, initial_weights: Optional[List[float]] = None):
        """
        Args:
            num_losses (int): Number of loss components to balance.
            tau (float): Temperature for softmax.
                         - tau > 1: weights are "softer" (closer to equal).
                         - 0 < tau < 1: weights are "harder" (more distinct).
                         - tau = 1: standard softmax.
            initial_weights (Optional[List[float]]): Initial weights. Sum must be 1.
                                                     If None, initialized equally.
        """
        super().__init__()
        assert num_losses > 0, "Number of losses must be positive"
        assert tau > 0, "Temperature (tau) must be positive"
        self.num_losses = num_losses
        self.tau = tau

        if initial_weights:
            assert len(initial_weights) == num_losses, \
                f"Number of initial weights ({len(initial_weights)}) must be equal to num_losses ({num_losses})"
            initial_weights_tensor = torch.tensor(initial_weights, dtype=torch.float32)
            assert torch.isclose(initial_weights_tensor.sum(), torch.tensor(1.0)), \
                "Sum of initial weights must be 1"
            # Initialize logit params from log of initial weights
            # so softmax(params) approximates initial_weights
            initial_params = torch.log(initial_weights_tensor)
        else:
            # Initializing with 0 yields equal weights after softmax
            initial_params = torch.zeros(num_losses, dtype=torch.float32)

        # 'params' are raw logits learned by optimizer
        self.params = nn.Parameter(initial_params)

    def forward(self, individual_losses: torch.Tensor) -> torch.Tensor:
        """
        Calculates total weighted loss.

        Args:
            individual_losses (torch.Tensor): 1D tensor containing loss values
                                              of each component.

        Returns:
            torch.Tensor: Total loss value (scalar).
        """
        if not isinstance(individual_losses, torch.Tensor):
            individual_losses = torch.stack(individual_losses)

        assert individual_losses.dim() == 1 and individual_losses.size(0) == self.num_losses, \
            f"Input individual_losses must be a 1D tensor of size {self.num_losses}"

        # 1. Calculate weights by applying softmax to learnable parameters
        weights = F.softmax(self.params / self.tau, dim=0)
        
        # 2. Calculate total loss by multiplying component losses with corresponding weights
        # Element-wise multiplication then sum (dot product)
        total_loss = torch.sum(weights * individual_losses)

        return total_loss

    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current weights for monitoring.
        These weights sum to 1.
        """
        with torch.no_grad():
            weights = F.softmax(self.params / self.tau, dim=0)
            return {f"weight_{i}": w.item() for i, w in enumerate(weights)}


class CombinedLoss(nn.Module):
    """
    Combined loss updated to use Focal Loss instead of CE Loss.
    Exact from notebook.
    """
    def __init__(self, 
                 num_classes=4, 
                 initial_loss_weights: Optional[List[float]] = None,
                 class_indices_for_rules: Optional[Dict[str, int]] = None):
        super().__init__()
        
        # --- Initialize loss components ---
        
        # 1. FOCAL LOSS
        self.fl = FocalLoss(gamma=2.0)
        print("Initialized with Focal Loss (gamma=2.0).")
        
        # 2. FOCAL TVERSKY LOSS - Exact values from notebook
        self.ftl = FocalTverskyLoss(
            num_classes=num_classes, 
            alpha=0.2,      # ← From notebook!
            beta=0.8,       # ← From notebook!
            gamma=4.0/3.0
        )
        print("Initialized with Focal Tversky Loss (alpha=0.2, beta=0.8, gamma=4/3).")

        # 3. Physics Loss
        self.pl = PhysicsLoss(scale_factor=0.01)
        
        # 4. Anatomical Rule Loss - ABLATION: DISABLED
        # if class_indices_for_rules is None:
        #     raise ValueError("`class_indices_for_rules` must be provided.")
        # self.arl = AnatomicalRuleLoss(class_indices=class_indices_for_rules)
        self.arl = None  # ABLATION STUDY: No Anatomical Loss
        print("⚠️  ABLATION STUDY: Anatomical Loss DISABLED")
        
        self.loss_weighter = DynamicLossWeighter(num_losses=3, initial_weights=initial_loss_weights)

    def forward(self, logits, targets, b1=None, all_es=None):
        # --- Calculate individual loss components ---
        l_fl = self.fl(logits, targets)  # Tính Focal Loss
        l_ftl = self.ftl(logits, targets)  # Tính Focal Tversky Loss

        lphy = torch.tensor(0.0, device=logits.device)
        if self.pl is not None and b1 is not None and all_es:
            try:
                e1, s1 = all_es[0]
                lphy = self.pl(b1, e1, s1)
            except (IndexError, TypeError):
                print("Warning: Physics loss skipped due to unexpected `all_es` format.")
        
        # ABLATION: No Anatomical Loss
        # larl = self.arl(logits) if self.arl else torch.tensor(0.0, device=logits.device)
        
        # --- Kết hợp 3 thành phần loss ---
        individual_losses = torch.stack([l_fl, l_ftl, lphy])
        total_loss = self.loss_weighter(individual_losses)
        
        return total_loss

    def get_current_loss_weights(self) -> Dict[str, float]:
        """Helper to monitor weights between loss functions."""
        weights = self.loss_weighter.get_current_weights()
        # Update names for clarity (3 components for ablation study)
        return {
            "weight_FocalLoss": weights["weight_0"],
            "weight_FocalTverskyLoss": weights["weight_1"],
            "weight_Physics": weights["weight_2"]
            # "weight_Anatomical": weights["weight_3"]  # ABLATION: REMOVED
        }
