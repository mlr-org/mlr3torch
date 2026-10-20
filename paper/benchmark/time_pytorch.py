import torch
import time
import math
from torch import nn
import numpy as np

# 3. Define the timing function
def time_pytorch(epochs, batch_size, n_layers, latent, n, p, device, seed, optimizer, jit, n_threads=1):
    torch.manual_seed(seed)
    torch.set_num_threads(int(n_threads))
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
        opt_class = torch.optim.AdamW
    elif optimizer == "sgd":
        opt_class = torch.optim.SGD
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, Y)
    # Load whole batches with a single indexing operation, just like the .getbatch() method
    # in R torch and mlr3torch, instead of indexing single observations and stacking them.
    sampler = torch.utils.data.BatchSampler(
        torch.utils.data.SequentialSampler(dataset), batch_size=batch_size, drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=None)

    steps = math.ceil(n / batch_size)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Like in R, every run creates a new optimizer, as mlr3torch does in every $train() call.
    def train_run(epochs):
        opt = opt_class(net.parameters(), lr = lr)
        sync()
        t0 = time.time()
        for _ in range(epochs):
            for (x, y) in dataloader:
                opt.zero_grad()
                y_hat = net(x)
                loss = loss_fn(y_hat, y)
                loss.backward()
                opt.step()
        sync()
        return time.time() - t0

    # warmup
    train_run(epochs = 4)
    t = train_run(epochs = epochs)

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

    return {'time': t, 'loss': mean_loss}


if __name__ == "__main__":
    # Called from the benchmark (benchmark.R) with the arguments as JSON; the result is printed as JSON on the last line.
    import json
    import sys
    print(json.dumps(time_pytorch(**json.loads(sys.argv[1]))))
