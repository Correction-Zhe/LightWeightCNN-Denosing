
import torch
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
from loss import combined_loss
from utils.warmup import create_warmup_plateau_schedulers


def train_model(
    model,
    dataloader,
    num_epochs,
    device,
    log_file,
    model_save_path,
    loss_curve_path,
    lambda_mse,
    lambda_grad,
    lambda_tv,
    lambda_edge,
    early_stop_threshold,
    theta,
    base_lr: float = 1e-4,
    warmup_epochs: int = 5,
    plateau_factor: float = 0.1,
    plateau_patience: int = 3
):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # Write initial log and print to console
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "w") as f:
        f.write(f"Training started at {start_time_str}\n")
        f.write(f"Num epochs: {num_epochs}, early_stop_threshold: {early_stop_threshold}\n")
        f.write(
            f"lambda_mse={lambda_mse}, lambda_grad={lambda_grad}, "
            f"lambda_tv={lambda_tv}, lambda_edge={lambda_edge}, theta={theta}\n"
        )
    print(f"Training started at {start_time_str}")
    print(f"Num epochs: {num_epochs}, early_stop_threshold: {early_stop_threshold}")
    print(
        f"lambda_mse={lambda_mse}, lambda_grad={lambda_grad}, "
        f"lambda_tv={lambda_tv}, lambda_edge={lambda_edge}, theta={theta}"
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(loss_curve_path), exist_ok=True)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    warmup_scheduler, plateau_scheduler = create_warmup_plateau_schedulers(
        optimizer,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience
    )

    loss_history = []
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = epoch_mse = epoch_grad = 0.0
        epoch_tv = epoch_edge = 0.0

        for ori_img, den_img in dataloader:
            ori_img, den_img = ori_img.to(device), den_img.to(device)
            optimizer.zero_grad()
            denoised, noise_pred = model(ori_img)
            total, loss_mse, loss_grad, loss_tv, loss_edge, theta_mean = combined_loss(
                denoised=denoised,
                noise_pred=noise_pred,
                ori_img=ori_img,
                den_img=den_img,
                lambda_mse=lambda_mse,
                lambda_grad=lambda_grad,
                lambda_tv=lambda_tv,
                lambda_smooth=lambda_edge,
                theta=theta
            )
            total.backward()
            optimizer.step()

            epoch_loss += total.item()
            epoch_mse += loss_mse.item()
            epoch_grad += loss_grad.item()
            epoch_tv += loss_tv.item()
            epoch_edge += loss_edge.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        rel_improve = 0 if epoch == 0 else (loss_history[-2] - avg_loss) / loss_history[-2]
        elapsed = time.time() - start_time

        msg = (
            f"Epoch {epoch+1}/{num_epochs} | Total={avg_loss:.6f} | "
            f"MSE={epoch_mse/len(dataloader):.6f} | Grad={epoch_grad/len(dataloader):.6f} | "
            f"TV={epoch_tv/len(dataloader):.6f} | Edge={epoch_edge/len(dataloader):.6f} | "
            f"LR={current_lr:.1e} | Improve={rel_improve:.6f} | Time={elapsed:.1f}s | "
            f"theta={float(theta):.6f}"
        )
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        if abs(rel_improve) < early_stop_threshold and epoch > 0:
            stop_msg = f"Early stopping at epoch {epoch+1} (improve={rel_improve:.6f})"
            print(stop_msg)
            with open(log_file, "a") as f:
                f.write(stop_msg + "\n")
            break

    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(loss_curve_path)
    plt.close()

    torch.save(model.state_dict(), model_save_path)
    print(f"weights have been saved to : {model_save_path}")
    print(f"loss curve have been saved to : {loss_curve_path}")
    print(f"logs have been saved to : {log_file}")

    return model


def test_model_folder(model, test_image_dir, device, output_dir, param_info=""):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    import cv2, numpy as np
    for file_name in os.listdir(test_image_dir):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        image_path = os.path.join(test_image_dir, file_name)
        with torch.no_grad():
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
            denoised, noise_pred = model(img_tensor)
            denoised = denoised.cpu().numpy().squeeze()
            noise_pred = noise_pred.cpu().numpy().squeeze()
            denoised = np.clip(denoised, 0, 255)
            noise_pred = np.clip(noise_pred, -255, 255)
            concat = cv2.hconcat([denoised, noise_pred])
            out_name = f"{param_info}{file_name}"
            cv2.imwrite(os.path.join(output_dir, out_name), concat)
            print(f"Processed {file_name} -> {out_name}")
