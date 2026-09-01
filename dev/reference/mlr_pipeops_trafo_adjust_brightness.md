# Adjust Brightness Transformation

Calls
[`torchvision::transform_adjust_brightness`](https://torchvision.mlverse.org/reference/transform_adjust_brightness.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

Being a `trafo_` operator, its `stages` parameter starts out as
`"both"`, so the transformation is applied both while training and when
predicting. Set `stages = "train"` to restrict it to training. See
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md)
for the `stages` parameter and this naming convention.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("trafo_adjust_brightness")

## Parameters

|  |  |  |  |  |
|----|----|----|----|----|
| Id | Type | Default | Levels | Range |
| brightness_factor | numeric | \- |  | \\\[0, \infty)\\ |
| stages | character | \- | train, predict, both | \- |
| affect_columns | untyped | selector_all() |  | \- |
