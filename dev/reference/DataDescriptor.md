# Data Descriptor

A data descriptor is a rather internal data structure used in the
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
data type. In essence it is an annotated
[`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html)
and a preprocessing graph (consisting mosty of
[`PipeOpModule`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md)
operators). The additional meta data (e.g. pointer, shapes) allows to
preprocess
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)s
in an
[`mlr3pipelines::Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
just like any (non-lazy) data types. The preprocessing is applied when
[`materialize()`](https://mlr3torch.mlr-org.com/dev/reference/materialize.md)
is called on the
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md).

To create a data descriptor, you can also use the
[`as_data_descriptor()`](https://mlr3torch.mlr-org.com/dev/reference/as_data_descriptor.md)
function.

## Details

While it would be more natural to define this as an S3 class, we opted
for an R6 class to avoid the usual trouble of serializing S3 objects. If
each row contained a DataDescriptor as an S3 class, this would copy the
object when serializing.

## See also

ModelDescriptor, lazy_tensor

## Public fields

- `dataset`:

  ([`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html))  
  The dataset.

- `graph`:

  ([`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html))  
  The preprocessing graph.

- `dataset_shapes`:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`))  
  The shapes of the output.

- `input_map`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The input map from the dataset to the preprocessing graph.

- `pointer`:

  (`character(2)`)  
  The output pointer.

- `pointer_shape`:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  The shape of the output indicated by `pointer`.

- `dataset_hash`:

  (`character(1)`)  
  Hash for the wrapped dataset.

- `hash`:

  (`character(1)`)  
  Hash for the data descriptor.

- `graph_input`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The input channels of the preprocessing graph (cached to save time).

- `pointer_shape_predict`:

  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)  
  Internal use only.

## Methods

### Public methods

- [`DataDescriptor$new()`](#method-DataDescriptor-new)

- [`DataDescriptor$print()`](#method-DataDescriptor-print)

- [`DataDescriptor$clone()`](#method-DataDescriptor-clone)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    DataDescriptor$new(
      dataset,
      dataset_shapes = NULL,
      graph = NULL,
      input_map = NULL,
      pointer = NULL,
      pointer_shape = NULL,
      pointer_shape_predict = NULL,
      clone_graph = TRUE
    )

#### Arguments

- `dataset`:

  ([`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html))  
  The torch dataset. It should return a named
  [`list()`](https://rdrr.io/r/base/list.html) of
  [`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  objects.

- `dataset_shapes`:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`))  
  The shapes of the output. Names are the elements of the list returned
  by the dataset. If the shape is not `NULL` (unknown, e.g. for images
  of different sizes) the first dimension must be `NA` to indicate the
  batch dimension.

- `graph`:

  ([`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html))  
  The preprocessing graph. If left `NULL`, no preprocessing is applied
  to the data and `input_map`, `pointer`, `pointer_shape`, and
  `pointer_shape_predict` are inferred in case the dataset returns only
  one element.

- `input_map`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Character vector that must have the same length as the input of the
  graph. Specifies how the data from the `dataset` is fed into the
  preprocessing graph.

- `pointer`:

  (`character(2)` \| `NULL`)  
  Points to an output channel within `graph`: Element 1 is the
  `PipeOp`'s id and element 2 is that `PipeOp`'s output channel.

- `pointer_shape`:

  ([`integer()`](https://rdrr.io/r/base/integer.html) \| `NULL`)  
  Shape of the output indicated by `pointer`.

- `pointer_shape_predict`:

  ([`integer()`](https://rdrr.io/r/base/integer.html) or `NULL`)  
  Internal use only. Used in a
  [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html) to
  anticipate possible mismatches between train and predict shapes.

- `clone_graph`:

  (`logical(1)`)  
  Whether to clone the preprocessing graph.

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Prints the object

#### Usage

    DataDescriptor$print(...)

#### Arguments

- `...`:

  (any)  
  Unused

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    DataDescriptor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Create a dataset
ds = dataset(
  initialize = function() self$x = torch_randn(10, 3, 3),
  .getitem = function(i) list(x = self$x[i, ]),
  .length = function() nrow(self$x)
)()
dd = DataDescriptor$new(ds, list(x = c(NA, 3, 3)))
dd
#> <DataDescriptor: 1 ops>
#> * dataset_shapes: [x: (NA,3,3)]
#> * input_map: (x) -> Graph
#> * pointer: nop.29b623.x.output
#> * shape: [(NA,3,3)]
# is the same as using the converter:
as_data_descriptor(ds, list(x = c(NA, 3, 3)))
#> <DataDescriptor: 1 ops>
#> * dataset_shapes: [x: (NA,3,3)]
#> * input_map: (x) -> Graph
#> * pointer: nop.29b623.x.output
#> * shape: [(NA,3,3)]
```
