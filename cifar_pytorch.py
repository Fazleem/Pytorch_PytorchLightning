import torch
import torch.nn as nn
import torchvision  # for datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# hyper parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# Dataset & Dataloader
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
# look at one batch of data, using iter we can see one batch of data


def imshow(img):
    img = img / 2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# get some training images
examples = iter(train_loader)
images, labels = examples.next()
print(images.shape, labels.shape)
# imshow(torchvision.utils.make_grid(images))

# for i in range(5):
#     plt.subplot(3, 2, i + 1)
#     plt.imshow(samples[i][0])
# plt.show()

channels = 3
out_channels = 6
kernel_size = 5
# padding = 2 * 2
# model building
class Cifar_ConvNet(nn.Module):
    def __init__(self):
        super(Cifar_ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=16, kernel_size=kernel_size
        )
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU().cuda()

    def forward(self, x):
        # 4*32*32 -> apply conv filter(5*5), padding 1, stride 0, out_channels=6 -> 6*28*28
        # 6*28*28 -> apply max pool stride = 2, filter =2 -> 6*14*14
        # 6*14*14 -> apply conv filter(5*5), padding 1, stride 0, out_channels=16 -> 16*10*10
        # 16*10*10 -> apply max pool stride = 2, filter =2 -> 16*5*5

        # conv layer followed by activation function (adding activ_functi doesn't change the size) followed by pooling layer
        output = self.pool1(F.relu(self.conv1(x)))
        output = self.pool1(F.relu(self.conv2(output)))
        output = output.view(-1, 16 * 5 * 5)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output


model = Cifar_ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
n_total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784 (-1 so that tensor can find automatically for us)
        images = images.to(device)
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

print("Finished Training")
# testing - we dont't want any gradients
with torch.no_grad():
    no_correct = 0
    no_samples = 0
    n_correct_class = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # torch returns value & index
        _, predictions = torch.max(outputs, 1)
        no_samples = no_samples + labels.shape[0]
        no_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            predction = predictions[i]
            if label == predction:
                n_correct_class[label] += 1
            n_class_samples[label] += 1

    accuracy = 100.0 * no_correct / no_samples
    print(f"accuracy = {accuracy}")

    # to find accuracy for each individual class
    for i in range(10):
        acc = 100.0 * n_correct_class[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {accuracy}")
