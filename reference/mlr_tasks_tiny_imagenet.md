# Tiny ImageNet Classification Task

Subset of the famous ImageNet dataset. The data is obtained from
[`torchvision::tiny_imagenet_dataset()`](https://torchvision.mlverse.org/reference/tiny_imagenet_dataset.html).

The underlying
[`DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html)
contains columns `"class"`, `"image"`, `"..row_id"`, `"split"`, where
the last column indicates whether the row belongs to the train,
validation or test set that are provided in torchvision.

There are no labels for the test rows, so by default, these observations
are inactive, which means that the task uses only 110000 of the 120000
observations that are defined in the underlying data backend.

## Construction

    tsk("tiny_imagenet")

## Download

The [task](https://mlr3.mlr-org.com/reference/Task.html)'s backend is a
[`DataBackendLazy`](https://mlr3torch.mlr-org.com/reference/mlr_backends_lazy.md)
which will download the data once it is requested. Other meta-data is
already available before that. You can cache these datasets by setting
the `mlr3torch.cache` option to `TRUE` or to a specific path to be used
as the cache directory.

## Properties

- Task type: “classif”

- Properties: “multiclass”

- Has Missings: no

- Target: “class”

- Features: “image”

- Data Dimension: 120000x4

## References

Deng, Jia, Dong, Wei, Socher, Richard, Li, Li-Jia, Li, Kai, Fei-Fei, Li
(2009). “Imagenet: A large-scale hierarchical image database.” In *2009
IEEE conference on computer vision and pattern recognition*, 248–255.
IEEE.

## Examples

``` r
task = tsk("tiny_imagenet")
```
