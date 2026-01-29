# Rotate Augmentation

Calls
[`torchvision::transform_rotate`](https://torchvision.mlverse.org/reference/transform_rotate.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("augment_rotate"")

## Parameters

|                |           |                |                      |                       |
|----------------|-----------|----------------|----------------------|-----------------------|
| Id             | Type      | Default        | Levels               | Range                 |
| angle          | untyped   | \-             |                      | \-                    |
| resample       | integer   | 0              |                      | \\(-\infty, \infty)\\ |
| expand         | logical   | FALSE          | TRUE, FALSE          | \-                    |
| center         | untyped   | NULL           |                      | \-                    |
| fill           | untyped   | NULL           |                      | \-                    |
| stages         | character | \-             | train, predict, both | \-                    |
| affect_columns | untyped   | selector_all() |                      | \-                    |
