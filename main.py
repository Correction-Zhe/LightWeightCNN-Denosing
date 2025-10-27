
from model import SimpleDnCNN
from utils.AugmentedTestDataset import AugmentedTestDataset
from train import train_model, test_model_folder
from torch.utils.data import DataLoader

import torch
import os
import datetime


def main():
    ori_dir = r"/IR_Denoising/GTVCNN/datasets/train"
    den_dir = r"/IR_Denoising/GTVCNN/datasets/blur"
    test_dir = r""

    N = len(os.listdir(ori_dir))
    print(f"The number of original images：{N}")

    dataset = AugmentedTestDataset(
        ori_dir=ori_dir,
        den_dir=den_dir,
        scales=(1.1, 1.0, 0.9),
        patch_size=450,
        num_per_image=1,
        do_flip=True,
        do_rotate=False
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = len(dataset)
    print(f"Augmented samples：{M} （= {N} × 3）")
    num_epochs = 50
    lambda_mse = 30.0
    lambda_grad = 5.0
    lambda_tv = 10.0
    lambda_smooth = 10.0
    theta = 0.10

    early_stop_threshold = 0.001
    base_lr = 1e-4
    warmup_epochs = 5
    plateau_factor = 0.01
    plateau_patience = 4

    depth, n_channels, kernel_size = 7, 32, 3
    model = SimpleDnCNN(depth, n_channels, image_channels=1, kernel_size=kernel_size).to(device)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"depth_{depth}_time_{now}_theta{theta:.3f}"
    base_dir = r""
    log_file = os.path.join(base_dir, "log_dir", f"{exp_name}.txt")
    model_save_path = os.path.join(base_dir, "model_dir", f"{exp_name}.pth")
    loss_curve_path = os.path.join(base_dir, "curve_dir", f"{exp_name}.png")
    test_output_dir = os.path.join(base_dir, "test_results", exp_name)

    trained = train_model(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        device=device,
        log_file=log_file,
        model_save_path=model_save_path,
        loss_curve_path=loss_curve_path,
        lambda_mse=lambda_mse,
        lambda_grad=lambda_grad,
        lambda_tv=lambda_tv,
        lambda_smooth=lambda_smooth,
        early_stop_threshold=early_stop_threshold,
        theta=theta,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience
    )

    test_model_folder(
        trained,
        test_dir,
        device,
        test_output_dir,
        param_info=exp_name + "_"
    )


if __name__ == '__main__':
    main()
