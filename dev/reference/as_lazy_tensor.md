# Convert to Lazy Tensor

Convert a object to a
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md).

## Usage

``` r
as_lazy_tensor(x, ...)

# S3 method for class 'dataset'
as_lazy_tensor(x, dataset_shapes = NULL, ids = NULL, ...)
```

## Arguments

- x:

  (any)  
  Object to convert to a
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)

- ...:

  (any)  
  Additional arguments passed to the method.

- dataset_shapes:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`))  
  The shapes of the output. Names are the elements of the list returned
  by the dataset. If the shape is not `NULL` (unknown, e.g. for images
  of different sizes) the first dimension must be `NA` to indicate the
  batch dimension.

- ids:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Which ids to include in the lazy tensor.

## Examples

``` r
iris_ds = dataset("iris",
  initialize = function() {
    self$iris = iris[, -5]
  },
  .getbatch = function(i) {
    list(x = torch_tensor(as.matrix(self$iris[i, ])))
  },
  .length = function() nrow(self$iris)
)()
# no need to specify the dataset shapes as they can be inferred from the .getbatch method
# only first 5 observations
as_lazy_tensor(iris_ds, ids = 1:5)
#> <ltnsr[len=5, shapes=(4)]>
# all observations
head(as_lazy_tensor(iris_ds))
#> <ltnsr[len=6, shapes=(4)]>

iris_ds2 = dataset("iris",
  initialize = function() self$iris = iris[, -5],
  .getitem = function(i) list(x = torch_tensor(as.numeric(self$iris[i, ]))),
  .length = function() nrow(self$iris)
)()
# if .getitem is implemented we cannot infer the shapes as they might vary,
# so we have to annotate them explicitly
as_lazy_tensor(iris_ds2, dataset_shapes = list(x = c(NA, 4L)))[1:5]
#> <ltnsr[len=5, shapes=(4)]>

# Convert a matrix
lt = as_lazy_tensor(matrix(rnorm(100), nrow = 20))
materialize(lt[1:5], rbind = TRUE)
#> torch_tensor
#> -1.4000  0.4682  0.0700  1.0743  1.9243
#>  0.2553  0.3630 -0.6391 -0.6651  1.2984
#> -2.4373 -1.3045 -0.0500  1.1140  0.7488
#> -0.0056  0.7378 -0.2515 -0.2459  0.5562
#>  0.6216  1.8885  0.4448 -1.1776 -0.5483
#> [ CPUFloatType{5,5} ]
```
