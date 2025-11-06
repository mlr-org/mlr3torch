# Create a Neural Network Layer

Retrieve a neural network layer from the
[`mlr_pipeops`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops.html)
dictionary.

## Usage

``` r
nn(.key, ...)
```

## Arguments

- .key:

  (`character(1)`)  

- ...:

  (any)  
  Additional parameters, constructor arguments or fields.

## Examples

``` r
po1 = po("nn_linear", id = "linear")
# is the same as:
po2 = nn("linear")
```
