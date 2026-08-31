# Convert a Graph to a Torch Learner

Converts a
[`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)
representing a deep learning pipeline into a
[`GraphLearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/GraphLearnerTorch.md).
The advantage over using
[`as_learner()`](https://mlr3.mlr-org.com/reference/as_learner.html) is
that the resulting learner exposes methods like `$dataset()` and fields
like `$network`.

## Usage

``` r
as_learner_torch(x, ...)

# S3 method for class 'Graph'
as_learner_torch(x, id = NULL, ...)

# S3 method for class 'PipeOp'
as_learner_torch(x, ...)

# S3 method for class 'GraphLearner'
as_learner_torch(x, id = x$id, ...)
```

## Arguments

- x:

  (any)  
  The object to convert, e.g. a
  [`Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html).

- ...:

  (any)  
  Unused.

- id:

  (`character(1)`)  
  The id of the learner. Defaults to `"<task_type>.graph"`.

## Value

[`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html)

## See also

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/dev/reference/TorchIngressToken.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/dev/reference/nn_graph.md),
[`pipeop_torch()`](https://mlr3torch.mlr-org.com/dev/reference/pipeop_torch.md)

Other Learner:
[`GraphLearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/GraphLearnerTorch.md),
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.mlp.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.module.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.tabm`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tabm.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)

## Examples

``` r
graph = po("scale") %>>%
  po("torch_ingress_num") %>>%
  nn("linear", out_features = 10) %>>%
  nn("relu") %>>%
  nn("head") %>>%
  po("torch_loss", "cross_entropy") %>>%
  po("torch_optimizer", "adam", lr = 0.1) %>>%
  po("torch_model_classif", epochs = 1, batch_size = 32)

learner = as_learner_torch(graph)
learner$id
#> [1] "classif.graph"
learner$param_set$set_values(linear.out_features = 20, torch_optimizer.lr = 0.01)

task = tsk("iris")
learner$train(task)
learner$network
#> An `nn_module` containing 163 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • module_list: <nn_module_list> #163 parameters
learner$predict(task)
#> 
#> ── <PredictionClassif> for 150 observations: ───────────────────────────────────
#>  row_ids     truth  response
#>        1    setosa    setosa
#>        2    setosa    setosa
#>        3    setosa    setosa
#>      ---       ---       ---
#>      148 virginica virginica
#>      149 virginica virginica
#>      150 virginica virginica

learner$dataset(task, "train")
#> <task_dataset>
#>   Inherits from: <dataset>
#>   Public:
#>     .getbatch: function (index) 
#>     .getitem: function (index) 
#>     .length: function () 
#>     all_features: Petal.Length Petal.Width Sepal.Length Sepal.Width Species
#>     batch_constructor: function (data, cache = NULL) 
#>     cache_lazy_tensors: FALSE
#>     clone: function (deep = FALSE) 
#>     feature_ingress_tokens: list
#>     initialize: function (task, feature_ingress_tokens, target_batchgetter = NULL) 
#>     load_state_dict: function (x, ..., .refer_to_state_dict = FALSE) 
#>     state_dict: function () 
#>     target_batchgetter: function (data) 
#>     task: TaskClassif, TaskSupervised, Task, R6
```
