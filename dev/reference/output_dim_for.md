# Network Output Dimension

Calculates the output dimension of a neural network for a given task
that is expected by
[mlr3torch](https://CRAN.R-project.org/package=mlr3torch). For
classification, this is the number of classes (unless it is a binary
classification task, where it is 1). For regression, it is 1.

This is an S3 generic and the single place where
[mlr3torch](https://CRAN.R-project.org/package=mlr3torch) decides how
many output neurons a task needs: it is what
[`PipeOpTorchHead`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_head.md)
and the
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)s
that build their own head ask. Adding a method for a new task type is
therefore the way to support it, see the "Supporting Other Task Types"
section of
[`PipeOpTorchHead`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_head.md).

## Usage

``` r
output_dim_for(x, ...)
```

## Arguments

- x:

  (any)  
  The task.

- ...:

  (any)  
  Additional arguments. Not used yet.

## Value

(`integer(1)`) The number of output neurons.

## See also

[`PipeOpTorchHead`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_head.md)
