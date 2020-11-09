import torch
import torch.nn as nn
import torchvision  # for datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

# hyper parameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001


# model building
class DigitNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_classes):
        super(DigitNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        images, labels = batch
        # 100, 1, 28, 28
        # 100, 784 (-1 so that tensor can find automatically for us)
        images = images.reshape(-1, 28 * 28)
        labels = labels

        # forward pass
        predict_outputs = self(images)
        loss = F.cross_entropy(predict_outputs, labels)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        return train_loader


if __name__ == "__main__":
    # set up a trainer
    trainer = Trainer(max_epochs=num_epochs, fast_dev_run=False)
    # set up model
    model = DigitNet(input_size, hidden_size, num_classes)
    # fit to trainer
    trainer.fit(model)