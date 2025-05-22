"""
Copied and adapted from the colab:
https://colab.research.google.com/drive/1qnZBatCPSnjDW4lLm-3wT39Xb0A7ONMX?authuser=3&usp=classroom_web#scrollTo=FOOLJokMJ2kE
by Prof. Fabrizio Silvestri
"""
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import matplotlib.pyplot as plt
from lightning_models import Encoder, Decoder, LitAutoEncoder


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)


def main():
    pl.seed_everything(42)

    autoencoder = LitAutoEncoder(Encoder(), Decoder())
    dm = MNISTDataModule()
    trainer = pl.Trainer(max_epochs=3,
                         callbacks=[TQDMProgressBar(refresh_rate=20)],)
    trainer.fit(model=autoencoder, datamodule=dm)
    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)

    # Get a single batch (one image)
    batch = next(iter(dm.train_dataloader()))
    original_batch = batch[0][0], batch[1][0]
    original_image = original_batch[0]

    # Use `predict_step` to reconstruct the image
    reconstructed_image = autoencoder.predict_step(original_batch, batch_idx=0).detach()

    # Reshape to 28x28
    original_image = original_image.view(28, 28).numpy()
    reconstructed_image = reconstructed_image.view(28, 28).numpy()

    # Plot the original and reconstructed images
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(original_image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(reconstructed_image, cmap="gray")
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")

    plt.show()


if __name__ == "__main__":
    main()