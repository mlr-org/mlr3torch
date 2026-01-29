# Convert to Data Descriptor

Converts the input to a
[`DataDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/DataDescriptor.md).

## Usage

``` r
as_data_descriptor(x, dataset_shapes, ...)
```

## Arguments

- x:

  (any)  
  Object to convert.

- dataset_shapes:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`))  
  The shapes of the output. Names are the elements of the list returned
  by the dataset. If the shape is not `NULL` (unknown, e.g. for images
  of different sizes) the first dimension must be `NA` to indicate the
  batch dimension.

- ...:

  (any)  
  Further arguments passed to the
  [`DataDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/DataDescriptor.md)
  constructor.

## Examples

``` r
ds = dataset("example",
  initialize = function() self$iris = iris[, -5],
  .getitem = function(i) list(x = torch_tensor(as.numeric(self$iris[i, ]))),
  .length = function() nrow(self$iris)
)()
as_data_descriptor(ds, list(x = c(NA, 4L)))
#> <DataDescriptor: 1 ops>
#> * dataset_shapes: [x: (NA,4)]
#> * input_map: (x) -> Graph
#> * pointer: nop.ba1e5a.x.output
#> * shape: [(NA,4)]

# if the dataset has a .getbatch method, the shapes are inferred
ds2 = dataset("example",
  initialize = function() self$iris = iris[, -5],
  .getbatch = function(i) list(x = torch_tensor(as.matrix(self$iris[i, ]))),
  .length = function() nrow(self$iris)
)()
as_data_descriptor(ds2)
#> <DataDescriptor: 1 ops>
#> * dataset_shapes: [x: (NA,4)]
#> * input_map: (x) -> Graph
#> * pointer: nop.a1f1e3.x.output
#> * shape: [(NA,4)]
```
