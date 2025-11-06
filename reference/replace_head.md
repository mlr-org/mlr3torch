# Replace the head of a network Replaces the head of the network with a linear layer with d_out classes.

Replace the head of a network Replaces the head of the network with a
linear layer with d_out classes.

## Usage

``` r
replace_head(network, d_out)
```

## Arguments

- network:

  ([`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  The network

- d_out:

  (`integer(1)`)  
  The number of output classes.
