from pathlib import Path

from latent_DDPM_generate import generate

# Generation of images for all Latent DDPM checkpoints

CHECKPOINT_DIR = Path("outputs/checkpoints")


def find_checkpoint(epoch: int):
    pattern = f"latent_1000_epoch{epoch}_*.pt"
    matches = sorted(CHECKPOINT_DIR.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(f"No checkpoint found for epoch {epoch}")

    if len(matches) > 1:
        print(f"Warning: multiple checkpoints found for epoch {epoch}. Using newest.")
    return matches[-1]


def generate_from_epochs(epochs, autoencoder_checkpoint,
    num_images=16,
    batch_size=8,
    latent_size=8,
    latent_channels=4,
    timesteps=1000):

    for epoch in epochs:
        checkpoint = find_checkpoint(epoch)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch}")
        print(checkpoint.name)
        print("=" * 60)

        generate(
            diffusion_checkpoint=str(checkpoint),
            autoencoder_checkpoint=autoencoder_checkpoint,
            num_images=num_images,
            batch_size=batch_size,
            latent_size=latent_size,
            latent_channels=latent_channels,
            timesteps=timesteps,
        )


def main():
    autoencoder_checkpoint = (
        "outputs/checkpoints/"
        "fl_autoencoder2_epoch100_20260624_000920.pt"
    )

    epochs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    generate_from_epochs(
        epochs=epochs,
        autoencoder_checkpoint=autoencoder_checkpoint,
        num_images=16,
        batch_size=8,
        latent_size=16,#8,
        latent_channels=16,#4,
        timesteps=1000,
    )

# 1000 = 0.402 s per image (new)
# 100 = 0.03 s per image
# 10 = 0.004 s per image

if __name__ == "__main__":
    main()