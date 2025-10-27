import torch
import torch.nn.functional as F
import math

def signed_log(x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x) + epsilon)
def compute_edge_smoothness(img: torch.Tensor, edge_mask: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    d2_h = img[..., 2:] - 2 * img[..., 1:-1] + img[..., :-2]
    d2_v = img[..., 2:, :] - 2 * img[..., 1:-1, :] + img[..., :-2, :]
    edge_mask_h = edge_mask[..., 1:-1]
    edge_mask_v = edge_mask[..., 1:-1, :]
    smoothness_h = torch.mean(edge_mask_h * d2_h.pow(2))
    smoothness_v = torch.mean(edge_mask_v * d2_v.pow(2))

    return smoothness_h + smoothness_v


def combined_loss(
        denoised: torch.Tensor,
        noise_pred: torch.Tensor,
        ori_img: torch.Tensor,
        den_img: torch.Tensor,
        lambda_mse: float,
        lambda_grad: float,
        lambda_tv: float,
        lambda_smooth: float = 0.8,
        theta: float = 0.1,
        epsilon: float = 1e-6,
        smooth_kernel_size: int = 3
) -> tuple:
    noise_target = ori_img - den_img
    d = signed_log(noise_pred, epsilon) - signed_log(noise_target, epsilon)
    loss_noise = torch.mean(torch.sqrt(d * d + epsilon * epsilon))
    grad_h_d = denoised[..., 1:] - denoised[..., :-1]
    grad_v_d = denoised[..., 1:, :] - denoised[..., :-1, :]
    abs_h = torch.abs(grad_h_d)
    abs_v = torch.abs(grad_v_d)

    if smooth_kernel_size > 1:
        if len(abs_h.shape) == 3:
            gaussian_kernel = torch.ones(1, 1, smooth_kernel_size, smooth_kernel_size,
                                         dtype=denoised.dtype, device=denoised.device)
            gaussian_kernel = gaussian_kernel / (smooth_kernel_size * smooth_kernel_size)
            abs_h_smooth = F.conv2d(abs_h.unsqueeze(1), gaussian_kernel,
                                    padding=smooth_kernel_size // 2).squeeze(1)
            abs_v_smooth = F.conv2d(abs_v.unsqueeze(1), gaussian_kernel,
                                    padding=smooth_kernel_size // 2).squeeze(1)
        else:
            num_channels = abs_h.shape[1]
            gaussian_kernel = torch.ones(num_channels, 1, smooth_kernel_size, smooth_kernel_size,
                                         dtype=denoised.dtype, device=denoised.device)
            gaussian_kernel = gaussian_kernel / (smooth_kernel_size * smooth_kernel_size)
            abs_h_smooth = F.conv2d(abs_h, gaussian_kernel,
                                    padding=smooth_kernel_size // 2, groups=num_channels)
            abs_v_smooth = F.conv2d(abs_v, gaussian_kernel,
                                    padding=smooth_kernel_size // 2, groups=num_channels)
    else:
        abs_h_smooth = abs_h
        abs_v_smooth = abs_v

    abs_h_pad = F.pad(abs_h_smooth, (0, 1, 0, 0), mode='replicate')
    abs_v_pad = F.pad(abs_v_smooth, (0, 0, 0, 1), mode='replicate')
    grad_mag = torch.sqrt(abs_h_pad.pow(2) + abs_v_pad.pow(2) + epsilon)
    theta_t = torch.tensor(theta, dtype=denoised.dtype, device=denoised.device)
    theta_sq = theta_t * theta_t
    Bx = torch.exp(- grad_h_d.pow(2) / (theta_sq + epsilon))
    By = torch.exp(- grad_v_d.pow(2) / (theta_sq + epsilon))
    grad_h_o = ori_img[..., 1:] - ori_img[..., :-1]
    grad_v_o = ori_img[..., 1:, :] - ori_img[..., :-1, :]
    w_h = 1.0 - Bx
    w_v = 1.0 - By
    loss_grad_h = torch.mean(w_h * (grad_h_d - grad_h_o).pow(2))
    loss_grad_v = torch.mean(w_v * (grad_v_d - grad_v_o).pow(2))
    loss_grad = loss_grad_h + loss_grad_v
    h = torch.mean(Bx * abs_h_smooth)
    v = torch.mean(By * abs_v_smooth)
    loss_tv = (math.pi * h + v) ** 2
    B_mag = torch.exp(- grad_mag.pow(2) / (theta_sq + epsilon))
    edge_mask = 1.0 - B_mag
    loss_smooth = compute_edge_smoothness(denoised, edge_mask, epsilon)

    total_loss = (
        lambda_mse * loss_noise +
        lambda_grad * loss_grad +
        lambda_tv * loss_tv +
        lambda_smooth * loss_smooth
    )

    theta_mean = theta_t
    return total_loss, loss_noise, loss_grad, loss_tv, loss_smooth, theta_mean
