# Reshaping Transformation

Reshapes the tensor according to the parameter `shape`, by calling
[`torch_reshape()`](https://torch.mlverse.org/docs/reference/torch_reshape.html).
This preprocessing function is applied batch-wise.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_preproc_torch.md).

## Parameters

- `shape` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The desired output shape. The first dimension is the batch dimension
  and should usually be `-1`.
