# Create a Dataset from a Task

Creates a torch
[dataset](https://torch.mlverse.org/docs/reference/dataset.html) from an
mlr3 [`Task`](https://mlr3.mlr-org.com/reference/Task.html). The
resulting dataset's `$.get_batch()` method returns a list with elements
`x`, `y` and `index`:

- `x` is a list with tensors, whose content is defined by the parameter
  `feature_ingress_tokens`.

- `y` is the target variable and its content is defined by the parameter
  `target_batchgetter`.

- `.index` is the index of the batch in the task's data.

The data is returned on the device specified by the parameter `device`.

## Usage

``` r
task_dataset(task, feature_ingress_tokens, target_batchgetter = NULL)
```

## Arguments

- task:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html))  
  The task for which to build the
  [dataset](https://torch.mlverse.org/docs/reference/dataset.html).

- feature_ingress_tokens:

  (named [`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchIngressToken`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md))  
  Each ingress token defines one item in the `$x` value of a batch with
  corresponding names.

- target_batchgetter:

  (`function(data, device)`)  
  A function taking in arguments `data`, which is a `data.table`
  containing only the target variable, and `device`. It must return the
  target as a torch
  [tensor](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  on the selected device.

## Value

[`torch::dataset`](https://torch.mlverse.org/docs/reference/dataset.html)

## Examples

``` r
task = tsk("iris")
sepal_ingress = TorchIngressToken(
  features = c("Sepal.Length", "Sepal.Width"),
  batchgetter = batchgetter_num,
  shape = c(NA, 2)
)
petal_ingress = TorchIngressToken(
  features = c("Petal.Length", "Petal.Width"),
  batchgetter = batchgetter_num,
  shape = c(NA, 2)
)
ingress_tokens = list(sepal = sepal_ingress, petal = petal_ingress)

target_batchgetter = function(data) {
  torch_tensor(data = data[[1L]], dtype = torch_float32())$unsqueeze(2)
}
dataset = task_dataset(task, ingress_tokens, target_batchgetter)
batch = dataset$.getbatch(1:10)
batch
#> $x
#> $x$sepal
#> torch_tensor
#>  5.1000  3.5000
#>  4.9000  3.0000
#>  4.7000  3.2000
#>  4.6000  3.1000
#>  5.0000  3.6000
#>  5.4000  3.9000
#>  4.6000  3.4000
#>  5.0000  3.4000
#>  4.4000  2.9000
#>  4.9000  3.1000
#> [ CPUFloatType{10,2} ]
#> 
#> $x$petal
#> torch_tensor
#>  1.4000  0.2000
#>  1.4000  0.2000
#>  1.3000  0.2000
#>  1.5000  0.2000
#>  1.4000  0.2000
#>  1.7000  0.4000
#>  1.4000  0.3000
#>  1.5000  0.2000
#>  1.4000  0.2000
#>  1.5000  0.1000
#> [ CPUFloatType{10,2} ]
#> 
#> 
#> $.index
#> torch_tensor
#>   1
#>   2
#>   3
#>   4
#>   5
#>   6
#>   7
#>   8
#>   9
#>  10
#> [ CPULongType{10} ]
#> 
#> $y
#> torch_tensor
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#>  1
#> [ CPUFloatType{10,1} ]
#> 
```
