# PipeOp Torch Classifier

Builds a torch classifier and trains it.

## Parameters

See
[`LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)

## Input and Output Channels

There is one input channel `"input"` that takes in `ModelDescriptor`
during traing and a `Task` of the specified `task_type` during
prediction. The output is `NULL` during training and a `Prediction` of
given `task_type` during prediction.

## State

A trained
[`LearnerTorchModel`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md).

## Internals

A
[`LearnerTorchModel`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md)
is created by calling
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_learner.md)
on the provided
[`ModelDescriptor`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md)
that is received through the input channel. Then the parameters are set
according to the parameters specified in `PipeOpTorchModel` and its
'\$train()` method is called on the [`Task`][mlr3::Task] stored in the [`ModelDescriptor\`\].

## See also

Other PipeOps:
[`mlr_pipeops_nn_adaptive_avg_pool1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool1d.md),
[`mlr_pipeops_nn_adaptive_avg_pool2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool2d.md),
[`mlr_pipeops_nn_adaptive_avg_pool3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool3d.md),
[`mlr_pipeops_nn_avg_pool1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool1d.md),
[`mlr_pipeops_nn_avg_pool2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool2d.md),
[`mlr_pipeops_nn_avg_pool3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool3d.md),
[`mlr_pipeops_nn_batch_norm1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm1d.md),
[`mlr_pipeops_nn_batch_norm2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm2d.md),
[`mlr_pipeops_nn_batch_norm3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm3d.md),
[`mlr_pipeops_nn_block`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_block.md),
[`mlr_pipeops_nn_celu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_celu.md),
[`mlr_pipeops_nn_conv1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv1d.md),
[`mlr_pipeops_nn_conv2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv2d.md),
[`mlr_pipeops_nn_conv3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv3d.md),
[`mlr_pipeops_nn_conv_transpose1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose1d.md),
[`mlr_pipeops_nn_conv_transpose2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose2d.md),
[`mlr_pipeops_nn_conv_transpose3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose3d.md),
[`mlr_pipeops_nn_dropout`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_dropout.md),
[`mlr_pipeops_nn_elu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_elu.md),
[`mlr_pipeops_nn_flatten`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_flatten.md),
[`mlr_pipeops_nn_ft_cls`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_ft_cls.md),
[`mlr_pipeops_nn_ft_transformer_block`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_ft_transformer_block.md),
[`mlr_pipeops_nn_geglu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_geglu.md),
[`mlr_pipeops_nn_gelu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_gelu.md),
[`mlr_pipeops_nn_glu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_glu.md),
[`mlr_pipeops_nn_hardshrink`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardshrink.md),
[`mlr_pipeops_nn_hardsigmoid`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardsigmoid.md),
[`mlr_pipeops_nn_hardtanh`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardtanh.md),
[`mlr_pipeops_nn_head`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_head.md),
[`mlr_pipeops_nn_identity`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_identity.md),
[`mlr_pipeops_nn_layer_norm`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_layer_norm.md),
[`mlr_pipeops_nn_leaky_relu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_leaky_relu.md),
[`mlr_pipeops_nn_linear`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_linear.md),
[`mlr_pipeops_nn_log_sigmoid`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_log_sigmoid.md),
[`mlr_pipeops_nn_max_pool1d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool1d.md),
[`mlr_pipeops_nn_max_pool2d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool2d.md),
[`mlr_pipeops_nn_max_pool3d`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool3d.md),
[`mlr_pipeops_nn_merge`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge.md),
[`mlr_pipeops_nn_merge_cat`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_cat.md),
[`mlr_pipeops_nn_merge_prod`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_prod.md),
[`mlr_pipeops_nn_merge_sum`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_sum.md),
[`mlr_pipeops_nn_prelu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_prelu.md),
[`mlr_pipeops_nn_reglu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_reglu.md),
[`mlr_pipeops_nn_relu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_relu.md),
[`mlr_pipeops_nn_relu6`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_relu6.md),
[`mlr_pipeops_nn_reshape`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_reshape.md),
[`mlr_pipeops_nn_rrelu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_rrelu.md),
[`mlr_pipeops_nn_selu`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_selu.md),
[`mlr_pipeops_nn_sigmoid`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_sigmoid.md),
[`mlr_pipeops_nn_softmax`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softmax.md),
[`mlr_pipeops_nn_softplus`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softplus.md),
[`mlr_pipeops_nn_softshrink`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softshrink.md),
[`mlr_pipeops_nn_softsign`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softsign.md),
[`mlr_pipeops_nn_squeeze`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_squeeze.md),
[`mlr_pipeops_nn_tanh`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tanh.md),
[`mlr_pipeops_nn_tanhshrink`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tanhshrink.md),
[`mlr_pipeops_nn_threshold`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_threshold.md),
[`mlr_pipeops_nn_tokenizer_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tokenizer_categ.md),
[`mlr_pipeops_nn_tokenizer_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tokenizer_num.md),
[`mlr_pipeops_nn_unsqueeze`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_unsqueeze.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`mlr_pipeops_torch_loss`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_loss.md),
[`mlr_pipeops_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model.md),
[`mlr_pipeops_torch_model_regr`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model_regr.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3pipelines::PipeOpLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_learner.html)
-\>
[`mlr3torch::PipeOpTorchModel`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model.md)
-\> `PipeOpTorchModelClassif`

## Methods

### Public methods

- [`PipeOpTorchModelClassif$new()`](#method-PipeOpTorchModelClassif-new)

- [`PipeOpTorchModelClassif$clone()`](#method-PipeOpTorchModelClassif-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)

------------------------------------------------------------------------

### Method `new()`

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

### Method `clone()`

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
  po("nn_head") %>>%
  po("torch_loss", "cross_entropy") %>>%
  po("torch_optimizer", "adam"))$train(tsk("iris"))[[1L]]

print(md)
#> <ModelDescriptor: 2 ops>
#> * Ingress:  torch_ingress_num.input: [(NA,4)]
#> * Task:  iris [classif]
#> * Callbacks:  N/A
#> * Optimizer:  Adaptive Moment Estimation
#> * Loss:  Cross Entropy
#> * pointer:  nn_head.output [(NA,3)]

# build the learner from the model descriptor and train it
po_model = po("torch_model_classif", batch_size = 50, epochs = 1)
po_model$train(list(md))
#> $output
#> NULL
#> 
po_model$state
#> $model
#> $network
#> An `nn_module` containing 15 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • module_list: <nn_module_list> #15 parameters
#> 
#> $internal_valid_scores
#> NULL
#> 
#> $loss_fn
#> list()
#> 
#> $optimizer
#> $optimizer$param_groups
#> $optimizer$param_groups[[1]]
#> $optimizer$param_groups[[1]]$params
#> [1] 1 2
#> 
#> $optimizer$param_groups[[1]]$lr
#> [1] 0.001
#> 
#> $optimizer$param_groups[[1]]$weight_decay
#> [1] 0
#> 
#> $optimizer$param_groups[[1]]$betas
#> [1] 0.900 0.999
#> 
#> $optimizer$param_groups[[1]]$eps
#> [1] 1e-08
#> 
#> $optimizer$param_groups[[1]]$amsgrad
#> [1] FALSE
#> 
#> 
#> 
#> $optimizer$state
#> $optimizer$state$`1`
#> $optimizer$state$`1`$exp_avg
#> torch_tensor
#>  0.7024  0.2504  0.7728  0.3120
#> -0.2130 -0.0702 -0.2059 -0.0596
#> -0.4894 -0.1801 -0.5668 -0.2525
#> [ CPUFloatType{3,4} ]
#> 
#> $optimizer$state$`1`$exp_avg_sq
#> torch_tensor
#> 0.01 *
#>  2.0399  0.2579  2.5042  0.4116
#>   0.1982  0.0213  0.2043  0.0236
#>   0.9821  0.1326  1.3210  0.2614
#> [ CPUFloatType{3,4} ]
#> 
#> $optimizer$state$`1`$max_exp_avg_sq
#> torch_tensor
#> [ CPUFloatType{0} ]
#> 
#> $optimizer$state$`1`$step
#> torch_tensor
#>  3
#> [ CPULongType{1} ]
#> 
#> 
#> $optimizer$state$`2`
#> $optimizer$state$`2`$exp_avg
#> torch_tensor
#>  0.1152
#> -0.0302
#> -0.0850
#> [ CPUFloatType{3} ]
#> 
#> $optimizer$state$`2`$exp_avg_sq
#> torch_tensor
#> 0.0001 *
#>  5.5873
#>  0.4762
#>  2.9579
#> [ CPUFloatType{3} ]
#> 
#> $optimizer$state$`2`$max_exp_avg_sq
#> torch_tensor
#> [ CPUFloatType{0} ]
#> 
#> $optimizer$state$`2`$step
#> torch_tensor
#>  3
#> [ CPULongType{1} ]
#> 
#> 
#> 
#> 
#> $epochs
#> [1] 1
#> 
#> $callbacks
#> named list()
#> 
#> $seed
#> [1] 415631006
#> 
#> $task_col_info
#> Key: <id>
#>              id    type                      levels
#>          <char>  <char>                      <list>
#> 1: Petal.Length numeric                      [NULL]
#> 2:  Petal.Width numeric                      [NULL]
#> 3: Sepal.Length numeric                      [NULL]
#> 4:  Sepal.Width numeric                      [NULL]
#> 5:      Species  factor setosa,versicolor,virginica
#> 
#> attr(,"class")
#> [1] "learner_torch_model" "list"               
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
#> $param_vals$num_interop_threads
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
#> Empty data.table (0 rows and 3 cols): stage,class,msg
#> 
#> $train_time
#> [1] 0.074
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
#> [1] ‘1.2.0’
#> 
#> $internal_tuned_values
#> named list()
#> 
#> $data_prototype
#> Empty data.table (0 rows and 5 cols): Species,Petal.Length,Petal.Width,Sepal.Length,Sepal.Width
#> 
#> $task_prototype
#> Empty data.table (0 rows and 5 cols): Species,Petal.Length,Petal.Width,Sepal.Length,Sepal.Width
#> 
#> $train_task
#> 
#> ── <TaskClassif> (150x5): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Target classes: setosa, versicolor, virginica
#> • Properties: multiclass
#> • Features (4):
#>   • dbl (4): Petal.Length, Petal.Width, Sepal.Length, Sepal.Width
#> 
#> attr(,"class")
#> [1] "learner_state" "list"         
```
