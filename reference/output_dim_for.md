# Network Output Dimension

Calculates the output dimension of a neural network for a given task
that is expected by mlr3torch. For classification, this is the number of
classes (unless it is a binary classification task, where it is 1). For
regression, it is 1.

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
