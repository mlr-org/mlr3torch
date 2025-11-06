# MNIST Image classification

Classic MNIST image classification.

The underlying
[`DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html)
contains columns `"label"`, `"image"`, `"row_id"`, `"split"`, where the
last column indicates whether the row belongs to the train or test set.

The first 60000 rows belong to the training set, the last 10000 rows to
the test set.

## Source

<https://torchvision.mlverse.org/reference/mnist_dataset.html>

## Construction

    tsk("mnist")

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

- Target: “label”

- Features: “image”

- Data Dimension: 70000x4

## References

Lecun, Y., Bottou, L., Bengio, Y., Haffner, P. (1998). “Gradient-based
learning applied to document recognition.” *Proceedings of the IEEE*,
**86**(11), 2278-2324.
[doi:10.1109/5.726791](https://doi.org/10.1109/5.726791) .

## Examples

``` r
task = tsk("mnist")
```
