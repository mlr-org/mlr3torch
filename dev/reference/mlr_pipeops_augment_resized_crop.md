# Resized Crop Augmentation

Calls
[`torchvision::transform_resized_crop`](https://torchvision.mlverse.org/reference/transform_resized_crop.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("augment_resized_crop"")

## Parameters

|                |           |                |                      |                       |
|----------------|-----------|----------------|----------------------|-----------------------|
| Id             | Type      | Default        | Levels               | Range                 |
| top            | integer   | \-             |                      | \\(-\infty, \infty)\\ |
| left           | integer   | \-             |                      | \\(-\infty, \infty)\\ |
| height         | integer   | \-             |                      | \\(-\infty, \infty)\\ |
| width          | integer   | \-             |                      | \\(-\infty, \infty)\\ |
| size           | untyped   | \-             |                      | \-                    |
| interpolation  | integer   | 2              |                      | \\\[0, 3\]\\          |
| stages         | character | \-             | train, predict, both | \-                    |
| affect_columns | untyped   | selector_all() |                      | \-                    |
