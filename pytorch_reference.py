import torch
import numpy as np

# create a random variables
# x = torch.rand(2, 2)
# y = torch.rand(2, 2)

# y.add_(x)
# print(x)
# print(y)
# z = torch.add(x, y)
# print(z)


# slicing operations

# x = torch.rand(4, 4)
# print(x)

# # extract all rows for column 0
# print(x[:, 0])

# # extract row 1 for all colums
# print(x[1, :])

# # get position at 1 value
# print(x[1, 1])

# # use .item() to get only 1 element in the tensor
# print(x[1, 1].item())

# # reshaping tensor
# a = torch.rand_like(x)
# print(a)
# y = a.view(16)
# print(y)

# # resize
# # suppose if you just know one dimension and not sure about the other dimension use -1
# # here you only know what the second dimension should be
# b = a.view(-1, 8)
# print(b.size())

# # convert numpy to torch tensors and viceversa
# a = torch.ones(6)
# print(a)
# b = a.numpy()
# print(type(b))

# # both tensor and ndarray point to same memory location in cpu

# d = np.ones(4)
# print(d)
# e = torch.from_numpy(d)
# print(type(e))


if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.rand(5, 4, device=device)
    y = torch.ones(5, 4, device=device)
    z = x + y
    # z = z.numpy()
    # here you cannot convert tensor to numpy in gpu
    # so convert into cpu first and then into numpy
    z = z.to("cpu").numpy()
    print(z)

    # whenever you want to optimize a variable, set the require_grad=True
    a = torch.randn(4, requires_grad=True)
    b = a + 2
    c = a * a * a

    # to calculate gradient of a variable
    c = c.mean()
    c.backward()  # dz/dx
    print(c)

    # this works if it is a scalar value, if there is no scalar value and if we need to apply
    # backprop with vectors then do the following
    # (jacobian matrix) * (Vector) = (backprop) --> chain rule
    d = torch.randn(4, requires_grad=True)
    e = d + 2
    f = d * d * d
    g = torch.tensor([0.1, 1.0, 0.01, 0.001], dtype=torch.float64)
    # to calculate gradient of a variable using backward()
    c = c.mean()
    # f.backward(g)  # dz/dx
    print(e)

    # how to prevent a variable from tracking or remove them from compitational graph

    # 1. d.requires_grad_(False)

    # 2. d = d.detach()

    # 3
    with torch.no_grad():
        e = e + 2
        print(e)
    # print(d)

    # after calculating the gradient in training step we need to make sure the gradient is set to zero again
    weights = torch.ones(4, requires_grad=True)
    for epoch in range(3):
        model_out = (weights * 3).sum()
        model_out.backward()
        print(weights.grad)

        weights.grad.zero_()
    # c.backward()
    # print(c)


"""
pipeline using Pytorch:
1. Load data, clean and get the X,Y, X-train,Y-train, x-test, y-test and convert them into tensors
2. Build a model - build a class which consists of the model you want to build and define the layers etc, also include forward step
3. Build a training loop which has does the following
    a) forward pass, compute prediction
    b) backward pass and calculate gradients
    c) Update weights
"""
""" 
epoch = 1 forward and backward pass of all training samples
batch_size = number of training samples in one forward and backward pass
num_iterations = number of passes, each pass using batch size number of samples

ex: 1000 samples, batch_size = 20 --> 50 iterations 1 epoch
"""


"""
If you have to do any transforms use 'torchvision.transforms() module which consists in number of classes that help in applying transforms"
"""

"""

Softmax & Cross Entropy

->Softmax squashes the o/p b/w 0 and 1
-> Higher the outcome lower the Cross Entropy Loss and Lower the outcome, higher the Loss

Using numpy - This is how it works:
1. Y should be one hot encoded
2. Predictions - good and bad
3. Cross entropy and Predict 

Using pytorch
nn.CrossEntropyLoss - applies both nn.LogSoftmax + nn.NLLLoss
No softmax in the last layer (donot implement ourselves)
Y-labels shouldn't be one hot encoded

# in Multiclass classification - no softmax layer at the end
# in binary classificaiton - implement sigmoid function (>0.5 belongs to class 1, else class 0) - and use Binary Entropy Loss


Activation functions:
1) torch.nn - softmax, linear, tanh, sigmoid, relu
2) torch,nn.functional - leakyRelu


----------------------
Pipeline for Deep Learning model
----------------------

1. DataLoader, Transformation
2. Multilayer neural net, activation function -- (Model)
3. Loss and Optimizer
4. Training Loop - batch training (feed forward, update weights, update weights)
5. Model Evaluation
6. GPU Support


Custom Dataset
---------------
torch.utils.data.Dataset is an abstract class of Dataset
Override __len__ and __getitem__
read csv in __init__ and reading of images or indetail in __getitem__
"""


class A(Dataset):

    # initialize your data and download
    # step 1: Download, read data
    def __init__(self):
        pass

    def __len__(self):
        return

    def __getitem__(self):
        return
