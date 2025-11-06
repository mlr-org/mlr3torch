# Normalization Transformation

Calls
[`torchvision::transform_normalize`](https://torchvision.mlverse.org/reference/transform_normalize.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("trafo_normalize"")

## Parameters

|                |           |                |                      |
|----------------|-----------|----------------|----------------------|
| Id             | Type      | Default        | Levels               |
| mean           | untyped   | \-             |                      |
| std            | untyped   | \-             |                      |
| stages         | character | \-             | train, predict, both |
| affect_columns | untyped   | selector_all() |                      |
