'''
This source code is from Hörst et al.
and slightly adjusted to fit the needs of the project.

'''



import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluation.utils import Sobel


class GradientLoss(nn.Module):
    """
    Class for the computation of the loss regarding the distance between the predicted and the ground truth gradient.
    """

    def __init__(self):
        super(GradientLoss, self).__init__()

        self.sobel = Sobel()

    def get_sobel_kernel(self, size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(self, hv):
        """For calculating gradient."""
        kernel_h, kernel_v = self.get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    def forward(self, images_h, images_v, gt_h, gt_v, focus_mask=None):

        focus_mask = (focus_mask[..., None]).float()  # assume input NHWP
        focus_mask = torch.cat([focus_mask, focus_mask], axis=-1)
        pred_hv = torch.stack([images_v, images_h], dim=-1)
        gt_hv = torch.stack([gt_v, gt_h], dim=-1)
        true_grad = self.get_gradient_hv(gt_hv)
        pred_grad = self.get_gradient_hv(pred_hv)

        loss = pred_grad - true_grad
        loss = focus_mask * (loss * loss)

        loss = loss.sum() / (focus_mask.sum() + 1.0e-8)
        return loss
