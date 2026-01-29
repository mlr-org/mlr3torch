# Resizing Transformation

Calls
[`torchvision::transform_resize`](https://torchvision.mlverse.org/reference/transform_resize.html),
see there for more information on the parameters. The preprocessing is
applied to the whole batch.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("trafo_resize"")

## Parameters

|                |           |                |                                                                                                                               |
|----------------|-----------|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| Id             | Type      | Default        | Levels                                                                                                                        |
| size           | untyped   | \-             |                                                                                                                               |
| interpolation  | character | 2              | Undefined, Bartlett, Blackman, Bohman, Box, Catrom, Cosine, Cubic, Gaussian, Hamming, [...](https://rdrr.io/r/base/dots.html) |
| stages         | character | \-             | train, predict, both                                                                                                          |
| affect_columns | untyped   | selector_all() |                                                                                                                               |
