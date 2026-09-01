# PipeOp Torch Classifier

Builds a torch classifier and trains it.

## Parameters

See
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)

## Input and Output Channels

There is one input channel `"input"` that takes in `ModelDescriptor`
during traing and a `Task` of the specified `task_type` during
prediction. The output is `NULL` during training and a `Prediction` of
given `task_type` during prediction.

## State

A trained
[`LearnerTorchModel`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md).

## Internals

A
[`LearnerTorchModel`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)
is created by calling
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/dev/reference/model_descriptor_to_learner.md)
on the provided
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md)
that is received through the input channel. Then the parameters are set
according to the parameters specified in `PipeOpTorchModel` and its
`$train()` method is called on the
[`Task`](https://mlr3.mlr-org.com/reference/Task.html) stored in the
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/dev/reference/ModelDescriptor.md).

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3pipelines::PipeOpLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_learner.html)
-\>
[`PipeOpTorchModel`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_model.md)
-\> `PipeOpTorchModelClassif`

## Methods

### Public methods

- [`PipeOpTorchModelClassif$new()`](#method-PipeOpTorchModelClassif-initialize)

- [`PipeOpTorchModelClassif$clone()`](#method-PipeOpTorchModelClassif-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorchModelClassif$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchModelClassif$new(id = "torch_model_classif", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchModelClassif$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchModelClassif$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# simple logistic regression

# configure the model descriptor
md = as_graph(po("torch_ingress_num") %>>%
  nn("head") %>>%
  po("torch_loss", "cross_entropy") %>>%
  po("torch_optimizer", "adam"))$train(tsk("iris"))[[1L]]

print(md)
#> <ModelDescriptor: 2 ops>
#> * Ingress:  torch_ingress_num.input: [(NA,4)]
#> * Task:  iris [classif]
#> * Callbacks:  N/A
#> * Optimizer:  Adaptive Moment Estimation
#> * Loss:  Cross Entropy
#> * pointer:  head.output [(NA,3)]

# build the learner from the model descriptor and train it
po_model = po("torch_model_classif", batch_size = 50, epochs = 1)
po_model$train(list(md))
#> $output
#> NULL
#> 
po_model$state
#> $model
#> <learner_torch_model> trained for 1 epoch
#> * Network:  <nn_graph> with 15 parameters
#> * Callbacks:  -
#> * Fields:  network, internal_valid_scores, loss_fn, optimizer, epochs,
#>   callbacks, seed, task_col_info
#> 
#> $param_vals
#> $param_vals$epochs
#> [1] 1
#> 
#> $param_vals$device
#> [1] "auto"
#> 
#> $param_vals$num_threads
#> [1] 1
#> 
#> $param_vals$seed
#> [1] "random"
#> 
#> $param_vals$eval_freq
#> [1] 1
#> 
#> $param_vals$measures_train
#> list()
#> 
#> $param_vals$measures_valid
#> list()
#> 
#> $param_vals$patience
#> [1] 0
#> 
#> $param_vals$min_delta
#> [1] 0
#> 
#> $param_vals$restore_best_weights
#> [1] FALSE
#> 
#> $param_vals$batch_size
#> [1] 50
#> 
#> $param_vals$shuffle
#> [1] TRUE
#> 
#> $param_vals$tensor_dataset
#> [1] FALSE
#> 
#> $param_vals$jit_trace
#> [1] FALSE
#> 
#> 
#> $log
#> Empty data.table (0 rows and 3 cols): stage,class,condition
#> 
#> $train_time
#> elapsed 
#>   0.066 
#> 
#> $task_hash
#> [1] "abc694dd29a7a8ce"
#> 
#> $feature_names
#> [1] "Petal.Length" "Petal.Width"  "Sepal.Length" "Sepal.Width" 
#> 
#> $validate
#> NULL
#> 
#> $mlr3_version
#> [1] ‘1.8.0’
#> 
#> $internal_tuned_values
#> named list()
#> 
#> $data_prototype
#> Empty data.table (0 rows and 5 cols): Species,Petal.Length,Petal.Width,Sepal.Length,Sepal.Width
#> 
#> $train_task
#> 
#> ── <TaskClassif> (150x5): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Properties: multiclass
#> • Features (4):
#>   • dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width
#> • Target classes: setosa, versicolor, virginica
#> 
#> attr(,"class")
#> [1] "learner_state" "list"         
```
