# Replace the Head of a Network

Replaces the last layer of a pretrained network with a fresh
[`torch::nn_linear`](https://torch.mlverse.org/docs/reference/nn_linear.html)
that has `d_out` output features, so a network trained on some other
task can be fine-tuned on this one. The new layer's input size is read
off the layer it replaces, and its weights are newly initialized while
the rest of the network keeps its pretrained weights.

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
