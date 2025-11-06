# Adjust Saturation Transformation

Calls
[`torchvision::transform_adjust_saturation`](https://torchvision.mlverse.org/reference/transform_adjust_saturation.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("trafo_adjust_saturation"")

## Parameters

|                   |           |                |                      |                       |
|-------------------|-----------|----------------|----------------------|-----------------------|
| Id                | Type      | Default        | Levels               | Range                 |
| saturation_factor | numeric   | \-             |                      | \\(-\infty, \infty)\\ |
| stages            | character | \-             | train, predict, both | \-                    |
| affect_columns    | untyped   | selector_all() |                      | \-                    |
