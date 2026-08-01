# TabM

TabM is an MLP-based tabular deep learning model that efficiently
represents an ensemble of `k` MLPs: the `k` submodels are trained in
parallel on the same batches and share most of their weights, which acts
as a strong regularizer. The network produces `k` predictions per
observation; the learner averages the predicted *probabilities*
(classification) or the predicted values (regression) over the `k`
submodels, and its loss function trains all `k` submodels jointly.

Numerical features are used as-is, or – if the `num_embeddings`
parameter is set – embedded feature-wise first, which usually improves
the performance considerably. Categorical features are one-hot encoded.

## Dictionary

This [Learner](https://mlr3.mlr-org.com/reference/Learner.html) can be
instantiated using the sugar function
[`lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html):

    lrn("classif.tabm", ...)
    lrn("regr.tabm", ...)

## Properties

- Supported task types: 'classif', 'regr'

- Predict Types:

  - classif: 'response', 'prob'

  - regr: 'response'

- Feature Types: “logical”, “integer”, “numeric”, “factor”, “ordered”

- Required Packages: [mlr3](https://CRAN.R-project.org/package=mlr3),
  [mlr3torch](https://CRAN.R-project.org/package=mlr3torch),
  [torch](https://CRAN.R-project.org/package=torch)

## Parameters

Parameters from
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
as well as:

- `arch_type` :: `character(1)`  
  The architecture type, one of:

  - `"tabm"` (default) – BatchEnsemble with the TabM initialization,
    i.e. all multiplicative adapters except the very first one are
    initialized with ones.

  - `"tabm-mini"` – all non-shared parameters are concentrated in a
    single elementwise affine transformation applied to the input.

- `k` :: `integer(1)`  
  The number of ensemble members. Default is `32`.

- `n_blocks` :: `integer(1)`  
  The number of blocks of the MLP backbone. If unset, `2` is used when
  `num_embeddings` is set and `3` otherwise.

- `d_block` :: `integer(1)`  
  The width of the MLP backbone. Default is `512`.

- `dropout` :: `numeric(1)`  
  The dropout rate. Default is `0.1`.

- `activation` :: `character(1)`, `nn_module_generator` or `function`  
  The activation function of the MLP backbone. Either the name of an
  activation of the `torch` package (e.g. `"relu"`, `"nn_relu"` or
  `"ReLU"`), an
  [`nn_module_generator`](https://torch.mlverse.org/docs/reference/nn_module.html)
  such as
  [`nn_relu`](https://torch.mlverse.org/docs/reference/nn_relu.html), or
  a function returning an
  [`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html).
  Default is `"relu"`.

- `start_scaling_init` :: `character(1)`  
  The initialization of the very first (non-shared) scaling, either
  `"random-signs"` or `"normal"`. If unset, `"normal"` is used when
  `num_embeddings` is set and `"random-signs"` otherwise.

Parameters of the embeddings for the numerical features:

- `num_embeddings` :: `character(1)`  
  The type of the numerical feature embeddings, one of `"none"`
  (default), `"linear_relu"`, `"periodic"` or `"piecewise_linear"`. The
  last two usually perform best.

- `d_embedding` :: `integer(1)`  
  The embedding size. If unset, `32` is used for `"linear_relu"`, `24`
  for `"periodic"` and `16` for `"piecewise_linear"`.

- `n_frequencies` :: `integer(1)`  
  `"periodic"` only: the number of frequencies per feature. Default is
  `48`.

- `frequency_init_scale` :: `numeric(1)`  
  `"periodic"` only: the initialization scale of the frequencies. This
  is an important hyperparameter. Default is `0.01`.

- `lite` :: `logical(1)`  
  `"periodic"` only: whether the outer linear layer is shared between
  all features. Default is `FALSE`.

- `embedding_activation` :: `logical(1)`  
  `"periodic"` and `"piecewise_linear"` only: whether a ReLU is applied
  at the end of the embedding. If unset, `TRUE` is used for `"periodic"`
  and `FALSE` for `"piecewise_linear"`.

- `n_bins` :: `integer(1)`  
  `"piecewise_linear"` only: the number of quantile bins, computed from
  the training data. Must be smaller than the number of training
  observations. Default is `48`.

## Loss and Prediction

The network output has shape `(batch, k, d_out)`. At training time the
learner therefore applies the configured loss to the `k` predictions
separately: the ensemble dimension is folded into the batch dimension
and each target is repeated `k` times. `$loss` itself is left untouched
and stays whatever was configured. For prediction, the per-submodel
probabilities (softmax for multiclass, sigmoid for binary) are averaged
over the `k` submodels; for regression the outputs are averaged.

## References

Gorishniy Y, Kotelnikov A, Babenko A (2025). “TabM: Advancing Tabular
Deep Learning with Parameter-Efficient Ensembling.” In *The Thirteenth
International Conference on Learning Representations (ICLR)*.
2410.24210, <https://openreview.net/forum?id=Sd4wYYOhmY>.

Wen Y, Tran D, Ba J (2020). “BatchEnsemble: An Alternative Approach to
Efficient Ensemble and Lifelong Learning.” In *The Eighth International
Conference on Learning Representations (ICLR)*. 2002.06715,
<https://openreview.net/forum?id=Sklf1yrYDr>.

## See also

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.mlp.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.module.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
-\> `LearnerTorchTabM`

## Methods

### Public methods

- [`LearnerTorchTabM$new()`](#method-LearnerTorchTabM-initialize)

- [`LearnerTorchTabM$clone()`](#method-LearnerTorchTabM-clone)

Inherited methods

- [`mlr3::Learner$base_learner()`](https://mlr3.mlr-org.com/reference/Learner.html#method-base_learner)
- [`mlr3::Learner$configure()`](https://mlr3.mlr-org.com/reference/Learner.html#method-configure)
- [`mlr3::Learner$encapsulate()`](https://mlr3.mlr-org.com/reference/Learner.html#method-encapsulate)
- [`mlr3::Learner$help()`](https://mlr3.mlr-org.com/reference/Learner.html#method-help)
- [`mlr3::Learner$predict()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict)
- [`mlr3::Learner$predict_newdata()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict_newdata)
- [`mlr3::Learner$reset()`](https://mlr3.mlr-org.com/reference/Learner.html#method-reset)
- [`mlr3::Learner$selected_features()`](https://mlr3.mlr-org.com/reference/Learner.html#method-selected_features)
- [`mlr3::Learner$train()`](https://mlr3.mlr-org.com/reference/Learner.html#method-train)
- [`LearnerTorch$dataset()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-dataset)
- [`LearnerTorch$format()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-format)
- [`LearnerTorch$marshal()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-marshal)
- [`LearnerTorch$print()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-print)
- [`LearnerTorch$unmarshal()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-unmarshal)

------------------------------------------------------------------------

### `LearnerTorchTabM$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    LearnerTorchTabM$new(
      task_type,
      optimizer = NULL,
      loss = NULL,
      callbacks = list()
    )

#### Arguments

- `task_type`:

  (`character(1)`)  
  The task type, either `"classif`" or `"regr"`.

- `optimizer`:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md))  
  The optimizer to use for training. Per default, *adam* is used.

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md))  
  The loss used to train the network. Per default, *mse* is used for
  regression and *cross_entropy* for classification.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md)s)  
  The callbacks. Must have unique ids.

------------------------------------------------------------------------

### `LearnerTorchTabM$clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchTabM$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Define the Learner and set parameter values
learner = lrn("classif.tabm")
learner$param_set$set_values(
  epochs = 1, batch_size = 16, device = "cpu",
  k = 4, n_blocks = 2, d_block = 32
)

# Define a Task
task = tsk("iris")

# Create train and test set
ids = partition(task)

# Train the learner on the training ids
learner$train(task, row_ids = ids$train)

# Make predictions for the test rows
predictions = learner$predict(task, row_ids = ids$test)

# Score the predictions
predictions$score()
#> classif.ce 
#>        0.6 
```
