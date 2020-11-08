import torch
import torch.nn as nn
import torchvision  # for datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# hyper parameters
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Dataset & Dataloader
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# look at one batch of data, using iter we can see one batch of data
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.imshow(samples[i][0])
# plt.show()


# model building
class DigitNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_classes):
        super(DigitNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size).cuda()
        self.relu = nn.ReLU().cuda()
        self.layer2 = nn.Linear(hidden_size, num_classes).cuda()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


model = DigitNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784 (-1 so that tensor can find automatically for us)
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        predict_outputs = model(images)
        loss = criterion(predict_outputs, labels)

        # backward pass
        optimizer.zero_grad()  # to empty the values in gradient attribute
        loss.backward()  # backward grad
        optimizer.step()  # update parameters

        # for every 100 step print the following
        if (i + 1) % 100 == 0:
            print(
                f"epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_step}, loss= {loss.item():.3f}"
            )
# testing - we dont't want any gradients
with torch.no_grad():
    no_correct = 0
    no_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # torch returns value & index
        _, predictions = torch.max(outputs, 1)
        no_samples = no_samples + labels.shape[0]
        no_correct += (predictions == labels).sum().item()

    accuracy = 100.0 * no_correct / no_samples
    print(f"accuracy = {accuracy}")
