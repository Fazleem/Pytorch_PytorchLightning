# warnings, ml tips, wrapper class

"""

Need not worry about the following

model.train()
model.eval()

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#easy to turn off gpu/tpu support
#easy to scale GPUs


# backward pass
optimizer.zero_grad()  # to empty the values in gradient attribute
loss.backward()  # backward grad
optimizer.step()  # update parameters

with torch.no_grad():
    pass

x=x.detach()

Bonus: Tensorboard support, prints tips/hints
"""