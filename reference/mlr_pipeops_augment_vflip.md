# Vertical Flip Augmentation

Calls
[`torchvision::transform_vflip`](https://torchvision.mlverse.org/reference/transform_vflip.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("augment_vflip"")

## Parameters

|                |           |                |                      |
|----------------|-----------|----------------|----------------------|
| Id             | Type      | Default        | Levels               |
| stages         | character | \-             | train, predict, both |
| affect_columns | untyped   | selector_all() |                      |
