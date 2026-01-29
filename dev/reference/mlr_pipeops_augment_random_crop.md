# Random Crop Augmentation

Calls
[`torchvision::transform_random_crop`](https://torchvision.mlverse.org/reference/transform_random_crop.html),
see there for more information on the parameters. The preprocessing is
applied to each element of a batch individually.

## Format

[`R6Class`](https://r6.r-lib.org/reference/R6Class.html) inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Construction

    po("augment_random_crop"")

## Parameters

|                |           |                |                                    |
|----------------|-----------|----------------|------------------------------------|
| Id             | Type      | Default        | Levels                             |
| size           | untyped   | \-             |                                    |
| padding        | untyped   | NULL           |                                    |
| pad_if_needed  | logical   | FALSE          | TRUE, FALSE                        |
| fill           | untyped   | 0L             |                                    |
| padding_mode   | character | constant       | constant, edge, reflect, symmetric |
| stages         | character | \-             | train, predict, both               |
| affect_columns | untyped   | selector_all() |                                    |
