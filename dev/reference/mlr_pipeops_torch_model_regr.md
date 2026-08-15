# Torch Regression Model

Builds a torch regression model and trains it.

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
-\> `PipeOpTorchModelRegr`

## Methods

### Public methods

- [`PipeOpTorchModelRegr$new()`](#method-PipeOpTorchModelRegr-initialize)

- [`PipeOpTorchModelRegr$clone()`](#method-PipeOpTorchModelRegr-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### `PipeOpTorchModelRegr$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchModelRegr$new(id = "torch_model_regr", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchModelRegr$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchModelRegr$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# simple linear regression

# build the model descriptor
md = as_graph(po("torch_ingress_num") %>>%
  nn("head") %>>%
  po("torch_loss", "mse") %>>%
  po("torch_optimizer", "adam"))$train(tsk("mtcars"))[[1L]]

print(md)
#> <ModelDescriptor: 2 ops>
#> * Ingress:  torch_ingress_num.input: [(NA,10)]
#> * Task:  mtcars [regr]
#> * Callbacks:  N/A
#> * Optimizer:  Adaptive Moment Estimation
#> * Loss:  Mean Squared Error
#> * pointer:  head.output [(NA,1)]

# build the learner from the model descriptor and train it
po_model = po("torch_model_regr", batch_size = 20, epochs = 1)
po_model$train(list(md))
#> $output
#> NULL
#> 
po_model$state
#> $model
#> <learner_torch_model> trained for 1 epoch
#> * Network:  <nn_graph> with 11 parameters
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
#> [1] 20
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
#>   0.062 
#> 
#> $task_hash
#> [1] "c7c4f02878d51895"
#> 
#> $feature_names
#>  [1] "am"   "carb" "cyl"  "disp" "drat" "gear" "hp"   "qsec" "vs"   "wt"  
#> 
#> $validate
#> NULL
#> 
#> $mlr3_version
#> [1] ‘1.7.1’
#> 
#> $internal_tuned_values
#> named list()
#> 
#> $data_prototype
#> Empty data.table (0 rows and 11 cols): mpg,am,carb,cyl,disp,drat...
#> 
#> $train_task
#> 
#> ── <TaskRegr> (32x11): Motor Trends ────────────────────────────────────────────
#> • Target: mpg
#> • Properties: -
#> • Features (10):
#>   • dbl (10): am, carb, cyl, disp, drat, gear, hp, qsec, vs, wt
#> 
#> attr(,"class")
#> [1] "learner_state" "list"         
```
