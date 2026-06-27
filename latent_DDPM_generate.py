import time
from tqdm import tqdm
from pathlib import Path
import torch
from torchvision.utils import save_image, make_grid
from Denoising import UNet
from Diffusion import GaussianDiffusion
from Autoencoder import Autoencoder
from Autoencoder2 import Autoencoder2
# This file is used to generate images via latent DDPM. Mainly intented to be used as export for the generate function

def timestamp():
    return time.strftime("%d_%H%M%S")

def denormalize(x):
    return ((x.clamp(-1, 1) + 1) / 2)

# Load Latent DDPM
def load_diffusion_model(checkpoint_path, device, latent_channels=4, base_channels=64, time_emb_dim=256):

    model = UNet(
        image_channels=latent_channels,
        base_channels=base_channels,
        time_emb_dim=time_emb_dim,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model



# Load Autoencoder
def load_autoencoder(checkpoint_path, device, latent_channels=4, base_channels=64):

    autoencoder = Autoencoder2(
        in_channels=3,
        latent_channels=latent_channels,
        base_channels=base_channels,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["model_state_dict"])
    autoencoder.eval()

    return autoencoder

# Generation
@torch.no_grad()
def generate(diffusion_checkpoint, autoencoder_checkpoint, num_images=16, batch_size=8,
    latent_size=8, latent_channels=4, timesteps=1000, schedule="linear", output_root="outputs/samples", grid_b=True,):

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    diffusion_model = load_diffusion_model(
        diffusion_checkpoint,
        device=device,
        latent_channels=latent_channels,
    )

    autoencoder = load_autoencoder(
        autoencoder_checkpoint,
        device=device,
        latent_channels=latent_channels,
    )

    diffusion = GaussianDiffusion(
        timesteps=timesteps,
        schedule=schedule,
        device=device,
    )

    # Create output directory
    checkpoint_name = Path(diffusion_checkpoint).stem
    run_name = "_".join(checkpoint_name.split("_")[:3]) # Assumes runID_ddpm_epochX structure

    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nSaving samples to:")
    print(run_dir)


    generated_images = []
    total_start = time.perf_counter()
    remaining = num_images
    stats = torch.load("outputs/checkpoints/latent_stats.pt")

    mean = stats["mean"].to(device) # Since we used normalization during training we have to de-normalize during gen.
    std = stats["std"].to(device)

    pbar = None
    if not grid_b:
        pbar = tqdm(
            total=num_images,
            desc="Generating",
            unit="img"
        )

    while remaining > 0:

        current_batch = min(batch_size, remaining)
        batch_start = time.perf_counter()

        # Sample latent
        latents = diffusion.sample(
            model=diffusion_model,
            image_size=latent_size,
            batch_size=current_batch,
            channels=latent_channels,
        )
        latents = latents * std + mean
        images = autoencoder.decode(latents)

        batch_time = (time.perf_counter() - batch_start)
        #print(f"Generated {current_batch} images in {batch_time:.2f}s")

        generated_images.append(images.cpu())
        remaining -= current_batch

        if pbar is not None:
            pbar.update(current_batch)

    if pbar is not None:
        pbar.close()

    total_time = (time.perf_counter() - total_start)

    generated_images = torch.cat(generated_images, dim=0)
    generated_images = denormalize(generated_images)

    # Save images - then form into gridsave
    for idx, image in enumerate(generated_images):
        save_image(image, run_dir / f"sample_{idx:04d}.png")

    if grid_b:
        grid = make_grid(generated_images, nrow=int(num_images ** 0.5), normalize=False)
        save_image(grid, run_dir / "grid.png")

    # Measure generation time
    avg_time = (total_time / num_images)
    print("\nGeneration complete")
    print(f"Images: {num_images}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average/image: {avg_time:.3f}s")

    return generated_images

def main():

    diffusion_checkpoint = (
        "outputs/checkpoints/latent_1000_epoch200_20260624_192438.pt"
    )

    autoencoder_checkpoint = (
        "outputs/checkpoints/"
        "fl_autoencoder2_epoch100_20260624_000920.pt"
    )

    generate(
        diffusion_checkpoint=diffusion_checkpoint,
        autoencoder_checkpoint=autoencoder_checkpoint,
        num_images=4000,
        batch_size=16,
        latent_size=16,
        latent_channels=16,#4,
        timesteps=1000,
        schedule = "linear",
        grid_b=False,
    )


if __name__ == "__main__":
    main()

# 1000 timesteps = 0.402 s per image
# 100 timesteps = 0.029 s per image
# 10 timesteps = 0.004 s per image