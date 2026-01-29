# Cross Entropy Loss

The `cross_entropy` loss function selects the multi-class
([`nn_cross_entropy_loss`](https://torch.mlverse.org/docs/reference/nn_cross_entropy_loss.html))
or binary
([`nn_bce_with_logits_loss`](https://torch.mlverse.org/docs/reference/nn_bce_with_logits_loss.html))
cross entropy loss based on the number of classes. Because of this,
there is a slight reparameterization of the loss arguments, see
*Parameters*.

## Parameters

- `class_weight`::
  [`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)  
  The class weights. For multi-class problems, this must be a
  `torch_tensor` of length `num_classes` (and is passed as argument
  `weight` to
  [`nn_cross_entropy_loss`](https://torch.mlverse.org/docs/reference/nn_cross_entropy_loss.html)).
  For binary problems, this must be a scalar (and is passed as argument
  `pos_weight` to
  [`nn_bce_with_logits_loss`](https://torch.mlverse.org/docs/reference/nn_bce_with_logits_loss.html)).

&nbsp;

- `ignore_index`:: `integer(1)`  
  Index of the class which to ignore and which does not contribute to
  the gradient. This is only available for multi-class loss.

- `reduction` :: `character(1)`  
  The reduction to apply. Is either `"mean"` or `"sum"` and passed as
  argument `reduction` to either loss function. The default is `"mean"`.

## Examples

``` r
loss = t_loss("cross_entropy")
# multi-class
multi_ce = loss$generate(tsk("iris"))
multi_ce
#> An `nn_module` containing 0 parameters.

# binary
binary_ce = loss$generate(tsk("sonar"))
binary_ce
#> An `nn_module` containing 0 parameters.
```
