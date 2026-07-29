# Extract the Batch Size for a Given Phase

A
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
uses the `batch_size` parameter for both training and prediction, unless
`batch_size_predict` is set, which then takes precedence during
prediction. This helper resolves the batch size for one phase and is
useful when overwriting the private `.dataloader()` method of a
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).

## Usage

``` r
get_batch_size(param_vals, phase)
```

## Arguments

- param_vals:

  (named [`list()`](https://rdrr.io/r/base/list.html))  
  The parameter values, containing `batch_size` and/or
  `batch_size_predict`.

- phase:

  (`character(1)`)  
  Either `"train"` or `"predict"`.

## Value

(`integer(1)` or `NULL`)  
The batch size for the given phase or `NULL` if none is set.

## Examples

``` r
get_batch_size(list(batch_size = 16), "train")
get_batch_size(list(batch_size = 16, batch_size_predict = 32), "predict")
get_batch_size(list(batch_size_predict = 32), "train")
```
