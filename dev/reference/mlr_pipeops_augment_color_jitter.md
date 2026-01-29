# Color Jitter Augmentation

Calls
[`torchvision::transform_color_jitter`](https://torchvision.mlverse.org/reference/transform_color_jitter.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("augment_color_jitter"")

## Parameters

|                |           |                |                      |                  |
|----------------|-----------|----------------|----------------------|------------------|
| Id             | Type      | Default        | Levels               | Range            |
| brightness     | numeric   | 0              |                      | \\\[0, \infty)\\ |
| contrast       | numeric   | 0              |                      | \\\[0, \infty)\\ |
| saturation     | numeric   | 0              |                      | \\\[0, \infty)\\ |
| hue            | numeric   | 0              |                      | \\\[0, \infty)\\ |
| stages         | character | \-             | train, predict, both | \-               |
| affect_columns | untyped   | selector_all() |                      | \-               |
