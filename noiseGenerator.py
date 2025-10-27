import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def generate_vertical_fpn(width, height, intensity=10, pattern_type='gaussian', seed=None):

    rng = np.random.RandomState(seed) if seed is not None else np.random

    if pattern_type == 'gaussian':
        column_noise = rng.normal(0, intensity, width)

    elif pattern_type == 'uniform':
        column_noise = rng.uniform(-intensity, intensity, width)

    elif pattern_type == 'periodic':
        x = np.arange(width)
        column_noise = 0.0
        column_noise += (intensity * 0.4) * np.sin(2 * np.pi * x / 3.7)
        column_noise += (intensity * 0.3) * np.sin(2 * np.pi * x / 7.3)
        column_noise += (intensity * 0.2) * np.sin(2 * np.pi * x / 11.1)
        column_noise += (intensity * 0.15) * np.sin(2 * np.pi * x / 5.9 + np.pi / 4)
        column_noise += (intensity * 0.1)  * np.sin(2 * np.pi * x / 13.7 + np.pi / 3)
        column_noise += rng.normal(0, intensity * 0.05, width)

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    fpn_pattern = np.tile(column_noise.reshape(1, -1), (height, 1))
    return fpn_pattern


def add_noise_to_image(image, fpn_intensity=10, gaussian_intensity=5,
                       pattern_type='gaussian', enable_fpn=True,
                       enable_gaussian=True, fpn_seed=None, gaussian_seed=None):
    height, width = image.shape
    noisy_image = image.astype(np.float32)

    modulation_factor = (noisy_image / 255.0) * 0.9 + 0.1

    if enable_fpn:
        fpn_pattern = generate_vertical_fpn(width, height, fpn_intensity, pattern_type, seed=fpn_seed)
        noisy_image = noisy_image + fpn_pattern * modulation_factor

    if enable_gaussian:
        rng = np.random.RandomState(gaussian_seed) if gaussian_seed is not None else np.random
        gaussian_noise = rng.normal(0, gaussian_intensity, (height, width))
        noisy_image = noisy_image + gaussian_noise * modulation_factor

    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def process_folder(input_folder, output_folder, fpn_intensity=10,
                   gaussian_intensity=5, pattern_type='gaussian',
                   enable_fpn=True, enable_gaussian=True,
                   image_extensions=None, fpn_seed=None, gaussian_seed=None):

    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    exts = {e.lower() for e in image_extensions}
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    image_files = [
        p for p in Path(input_folder).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    image_files.sort(key=lambda p: p.name.lower())

    print(f" {len(image_files)} images have been detected")


    for idx, img_path in enumerate(tqdm(image_files, desc="处理图像")):
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning:can not read the images: {img_path}")
            continue

        noisy_image = add_noise_to_image(
            image,
            fpn_intensity=fpn_intensity,
            gaussian_intensity=gaussian_intensity,
            pattern_type=pattern_type,
            enable_fpn=enable_fpn,
            enable_gaussian=enable_gaussian,
            fpn_seed=fpn_seed,
            gaussian_seed=gaussian_seed
        )

        output_path = Path(output_folder) / img_path.name
        cv2.imwrite(str(output_path), noisy_image)

    print(f"Noisy images have been stored to : {output_folder}")



if __name__ == "__main__":
    FPN_SEED = 15
    GAUSSIAN_SEED = 15

    input_folder = r"E:\PycharmProjects\GTVCNN\LightWeightCNN_github\datasets\train\clean"
    output_folder = r"E:\PycharmProjects\GTVCNN\LightWeightCNN_github\datasets\train\noisy"

    process_folder(
        input_folder,
        output_folder,
        fpn_intensity=10,
        gaussian_intensity=0,
        pattern_type='gaussian',
        enable_fpn=True,
        enable_gaussian=True,
        fpn_seed=FPN_SEED,
        gaussian_seed=GAUSSIAN_SEED
    )
