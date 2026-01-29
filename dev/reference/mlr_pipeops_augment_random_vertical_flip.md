# Random Vertical Flip Augmentation

Calls
[`torchvision::transform_random_vertical_flip`](https://torchvision.mlverse.org/reference/transform_random_vertical_flip.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("augment_random_vertical_flip"")

## Parameters

|                |           |                |                      |              |
|----------------|-----------|----------------|----------------------|--------------|
| Id             | Type      | Default        | Levels               | Range        |
| p              | numeric   | 0.5            |                      | \\\[0, 1\]\\ |
| stages         | character | \-             | train, predict, both | \-           |
| affect_columns | untyped   | selector_all() |                      | \-           |
