# Create a Neural Network Layer

Retrieve a neural network layer from the
[`mlr_pipeops`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops.html)
dictionary.

The id of the returned
[`PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html) is
`.key`. Because the ids within a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) must
be unique, `.key` can be suffixed with `_<n>` to disambiguate repeated
layers, e.g. `nn("linear_1")` and `nn("linear_2")`.

## Usage

``` r
nn(.key, ...)
```

## Arguments

- .key:

  (`character(1)`)  
  The key of the layer in the dictionary, optionally followed by a
  `_<n>` suffix.

- ...:

  (any)  
  Additional parameters, constructor arguments or fields.

## Examples

``` r
po1 = po("nn_linear", id = "linear")
# is the same as:
po2 = nn("linear")

# the `_<n>` suffix is part of the id, but not of the dictionary key
nn("linear_1")$id
#> [1] "linear_1"
```
