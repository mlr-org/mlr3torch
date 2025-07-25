import torch
import time
import math
from torch import nn
import numpy as np

# 3. Define the timing function
def time_pytorch(epochs, batch_size, n_layers, latent, n, p, device, seed, optimizer, jit):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # convert latentn, n_layers to int
    latent = int(latent)
    n_layers = int(n_layers)
    p = int(p)
    # 2. Define a function to create the neural network
    def make_network(p, latent, n_layers):
        if n_layers == 0:
            return nn.Linear(p, 1)
        layers = [nn.Linear(p, latent), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(latent, latent))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(latent, 1))
        #return layers
        return nn.Sequential(*layers)

    device = torch.device(device)

    X = torch.randn(n, p, device=device)
    beta = torch.randn(p, 1, device=device)
    Y = X.matmul(beta) + torch.randn(n, 1, device=device) * 0.01

    # Create the network
    try:
        net = make_network(p, latent, n_layers)
    except Exception as e:
        print(f"Error occurred while creating the network: {e}")

    if jit:
        net = torch.jit.script(net)

    net.to(device)

    lr = 0.0001

    # Define optimizer and loss function
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr = lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr = lr)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    steps = math.ceil(n / batch_size)

    def train_run(epochs):
        for _ in range(epochs):
            for (x, y) in dataloader:
                optimizer.zero_grad()
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                optimizer.step()


    train_run(epochs = 5)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    train_run(epochs = epochs)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time() - t0

    net.eval()

    # evaluate the training loss without grad tracking and calculate the mean
    mean_loss = 0
    with torch.no_grad():
        # iterate over the dataset
        for (x, y) in dataloader:
            y_hat = net(x)
            loss = loss_fn(y_hat, y)
            mean_loss += loss.item()
    mean_loss /= steps

    # Get peak reserved bytes
    # for some reason we need to convert to float as otherwise we have some
    # type conversion issues from python -> R
    if device == "cuda":
        memory = float(torch.cuda.memory_reserved())
    else:
        memory = None


    return {'time': t, 'loss': mean_loss, 'memory': memory}


if __name__ == "__main__":
    print(time_pytorch(epochs=1, batch_size=32, n_layers=1, latent=1, n=2000, p=1000, device='cpu', seed=42, optimizer="sgd", jit = True))
