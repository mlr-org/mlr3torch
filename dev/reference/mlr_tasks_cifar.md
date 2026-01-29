# CIFAR Classification Tasks

The CIFAR-10 and CIFAR-100 datasets. A subset of the 80 million tiny
images dataset with noisy labels was supplied to student labelers, who
were asked to filter out incorrectly labeled images. The images are have
datatype
[`torch_long()`](https://torch.mlverse.org/docs/reference/torch_dtype.html).

CIFAR-10 contains 10 classes. CIFAR-100 contains 100 classes, which may
be partitioned into 20 superclasses of 5 classes each. The CIFAR-10 and
CIFAR-100 classes are mutually exclusive. See Chapter 3.1 of [the
technical
report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
for more details.

The data is obtained from
[`torchvision::cifar10_dataset()`](https://torchvision.mlverse.org/reference/cifar_datasets.html)
(or
[`torchvision::cifar100_dataset()`](https://torchvision.mlverse.org/reference/cifar_datasets.html)).

## Format

[R6::R6Class](https://r6.r-lib.org/reference/R6Class.html) inheriting
from
[mlr3::TaskClassif](https://mlr3.mlr-org.com/reference/TaskClassif.html).

## Construction

    tsk("cifar10")
    tsk("cifar100")

## Download

The [task](https://mlr3.mlr-org.com/reference/Task.html)'s backend is a
[`DataBackendLazy`](https://mlr3torch.mlr-org.com/dev/reference/mlr_backends_lazy.md)
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

- Data Dimension: 60000x4

## References

Krizhevsky, Alex (2009). “Learning Multiple Layers of Features from Tiny
Images.” *Master's thesis, Department of Computer Science, University of
Toronto*.

## Examples

``` r
task_cifar10 = tsk("cifar10")
task_cifar100 = tsk("cifar100")
```
