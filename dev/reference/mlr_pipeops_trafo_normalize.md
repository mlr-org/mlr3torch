# Normalization Transformation

Normalizes the channels by subtracting `mean` and dividing by `std`. The
channels are the third-to-last dimension, e.g. `C` for images of shape
`(N, C, H, W)` or videos of shape `(N, T, C, H, W)`, as in
[`torchvision::transform_normalize`](https://torchvision.mlverse.org/reference/transform_normalize.html),
which this is the same as for these inputs. For inputs with fewer
dimensions, e.g. `(N, C)` or `(N, C, L)`, the channels are the second
dimension. Inputs that are not floating point, e.g. images stored as
integers, are converted to the default floating point type
([`torch_get_default_dtype()`](https://torch.mlverse.org/docs/reference/default_dtype.html))
first. The preprocessing is applied to the whole batch.

Being a `trafo_` operator, its `stages` parameter starts out as
`"both"`, so the transformation is applied both while training and when
predicting. Set `stages = "train"` to restrict it to training. See
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md)
for the `stages` parameter and this naming convention.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("trafo_normalize")

## Parameters

|                |           |                |                      |
|----------------|-----------|----------------|----------------------|
| Id             | Type      | Default        | Levels               |
| mean           | untyped   | \-             |                      |
| std            | untyped   | \-             |                      |
| stages         | character | \-             | train, predict, both |
| affect_columns | untyped   | selector_all() |                      |
