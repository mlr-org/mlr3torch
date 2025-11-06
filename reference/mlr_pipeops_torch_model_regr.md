# Torch Regression Model

Builds a torch regression model and trains it.

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
[`mlr_pipeops_torch_model_classif`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model_classif.md)

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`mlr3pipelines::PipeOpLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_pipeops_learner.html)
-\>
[`mlr3torch::PipeOpTorchModel`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_model.md)
-\> `PipeOpTorchModelRegr`

## Methods

### Public methods

- [`PipeOpTorchModelRegr$new()`](#method-PipeOpTorchModelRegr-new)

- [`PipeOpTorchModelRegr$clone()`](#method-PipeOpTorchModelRegr-clone)

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

### Method `clone()`

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
  po("nn_head") %>>%
  po("torch_loss", "mse") %>>%
  po("torch_optimizer", "adam"))$train(tsk("mtcars"))[[1L]]

print(md)
#> <ModelDescriptor: 2 ops>
#> * Ingress:  torch_ingress_num.input: [(NA,10)]
#> * Task:  mtcars [regr]
#> * Callbacks:  N/A
#> * Optimizer:  Adaptive Moment Estimation
#> * Loss:  Mean Squared Error
#> * pointer:  nn_head.output [(NA,1)]

# build the learner from the model descriptor and train it
po_model = po("torch_model_regr", batch_size = 20, epochs = 1)
po_model$train(list(md))
#> $output
#> NULL
#> 
po_model$state
#> $model
#> $network
#> An `nn_module` containing 11 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • module_list: <nn_module_list> #11 parameters
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
#> -9.0099 -74.1069 -168.1532 -6699.1582 -90.6256 -92.6180 -4010.4956 -458.2927 -9.2334 -87.6529
#> [ CPUFloatType{1,10} ]
#> 
#> $optimizer$state$`1`$exp_avg_sq
#> torch_tensor
#> Columns 1 to 6 4.5227e+00  3.1500e+02  1.5743e+03  2.4867e+06  4.5644e+02  4.7793e+02
#> 
#> Columns 7 to 10 9.1484e+05  1.1628e+04  4.7174e+00  4.2677e+02
#> [ CPUFloatType{1,10} ]
#> 
#> $optimizer$state$`1`$max_exp_avg_sq
#> torch_tensor
#> [ CPUFloatType{0} ]
#> 
#> $optimizer$state$`1`$step
#> torch_tensor
#>  2
#> [ CPULongType{1} ]
#> 
#> 
#> $optimizer$state$`2`
#> $optimizer$state$`2`$exp_avg
#> torch_tensor
#> -25.8787
#> [ CPUFloatType{1} ]
#> 
#> $optimizer$state$`2`$exp_avg_sq
#> torch_tensor
#>  37.1321
#> [ CPUFloatType{1} ]
#> 
#> $optimizer$state$`2`$max_exp_avg_sq
#> torch_tensor
#> [ CPUFloatType{0} ]
#> 
#> $optimizer$state$`2`$step
#> torch_tensor
#>  2
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
#> [1] 177443662
#> 
#> $task_col_info
#>         id    type levels
#>     <char>  <char> <list>
#>  1:     am numeric [NULL]
#>  2:   carb numeric [NULL]
#>  3:    cyl numeric [NULL]
#>  4:   disp numeric [NULL]
#>  5:   drat numeric [NULL]
#>  6:   gear numeric [NULL]
#>  7:     hp numeric [NULL]
#>  8:   qsec numeric [NULL]
#>  9:     vs numeric [NULL]
#> 10:     wt numeric [NULL]
#> 11:    mpg numeric [NULL]
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
#> Empty data.table (0 rows and 3 cols): stage,class,msg
#> 
#> $train_time
#> [1] 0.046
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
#> [1] ‘1.2.0’
#> 
#> $internal_tuned_values
#> named list()
#> 
#> $data_prototype
#> Empty data.table (0 rows and 11 cols): mpg,am,carb,cyl,disp,drat...
#> 
#> $task_prototype
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
