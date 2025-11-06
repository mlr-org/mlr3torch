# Learner Torch Module

Create a torch learner from a torch module.

## Dictionary

This [Learner](https://mlr3.mlr-org.com/reference/Learner.html) can be
instantiated using the sugar function
[`lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html):

    lrn("classif.module", ...)
    lrn("regr.module", ...)

## Properties

- Supported task types: 'classif', 'regr'

- Predict Types:

  - classif: 'response', 'prob'

  - regr: 'response'

- Feature Types: “logical”, “integer”, “numeric”, “character”, “factor”,
  “ordered”, “POSIXct”, “Date”, “lazy_tensor”

- Required Packages: [mlr3](https://CRAN.R-project.org/package=mlr3),
  [mlr3torch](https://CRAN.R-project.org/package=mlr3torch),
  [torch](https://CRAN.R-project.org/package=torch)

## See also

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/reference/mlr_learners.mlp.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md)

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/reference/mlr_learners.mlp.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3torch::LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)
-\> `LearnerTorchModule`

## Methods

### Public methods

- [`LearnerTorchModule$new()`](#method-LearnerTorchModule-new)

- [`LearnerTorchModule$clone()`](#method-LearnerTorchModule-clone)

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
- [`mlr3torch::LearnerTorch$dataset()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-dataset)
- [`mlr3torch::LearnerTorch$format()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-format)
- [`mlr3torch::LearnerTorch$marshal()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-marshal)
- [`mlr3torch::LearnerTorch$print()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-print)
- [`mlr3torch::LearnerTorch$unmarshal()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-unmarshal)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    LearnerTorchModule$new(
      module_generator = NULL,
      param_set = NULL,
      ingress_tokens = NULL,
      task_type,
      properties = NULL,
      optimizer = NULL,
      loss = NULL,
      callbacks = list(),
      packages = character(0),
      feature_types = NULL,
      predict_types = NULL
    )

#### Arguments

- `module_generator`:

  (`function` or `nn_module_generator`)  
  A `nn_module_generator` or `function` returning an `nn_module`. Both
  must take as argument the `task` for which to construct the network.
  Other arguments to its initialize method can be provided as
  parameters.

- `param_set`:

  (`NULL` or
  [`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  If provided, contains the parameters for the module_generator. If
  `NULL`, parameters will be inferred from the module_generator.

- `ingress_tokens`:

  (`list` of
  [`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md))  
  A list with ingress tokens that defines how the dataset will be
  defined. The names must correspond to the arguments of the network's
  forward method. For numeric, categorical, and lazy tensor features,
  you can use
  [`ingress_num()`](https://mlr3torch.mlr-org.com/reference/ingress_num.md),
  [`ingress_categ()`](https://mlr3torch.mlr-org.com/reference/ingress_categ.md),
  and
  [`ingress_ltnsr()`](https://mlr3torch.mlr-org.com/reference/ingress_ltnsr.md)
  to create them.

- `task_type`:

  (`character(1)`)  
  The task type, either `"classif`" or `"regr"`.

- `task_type`:

  (`character(1)`)  
  The task type.

- `properties`:

  (`NULL` or [`character()`](https://rdrr.io/r/base/character.html))  
  The properties of the learner. Defaults to all available properties
  for the given task type.

- `optimizer`:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md))  
  The optimizer to use for training. Per default, *adam* is used.

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md))  
  The loss used to train the network. Per default, *mse* is used for
  regression and *cross_entropy* for classification.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)s)  
  The callbacks. Must have unique ids.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `feature_types`:

  (`NULL` or [`character()`](https://rdrr.io/r/base/character.html))  
  The feature types. Defaults to all available feature types.

- `predict_types`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The predict types. See
  [`mlr_reflections$learner_predict_types`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
  for available values.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchModule$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
nn_one_layer = nn_module("nn_one_layer",
  initialize = function(task, size_hidden) {
    self$first = nn_linear(task$n_features, size_hidden)
    self$second = nn_linear(size_hidden, output_dim_for(task))
  },
  # argument x corresponds to the ingress token x
  forward = function(x) {
    x = self$first(x)
    x = nnf_relu(x)
    self$second(x)
  }
)
learner = lrn("classif.module",
  module_generator = nn_one_layer,
  ingress_tokens = list(x = ingress_num()),
  epochs = 10,
  size_hidden = 20,
  batch_size = 16
)
task = tsk("iris")
learner$train(task)
learner$network
#> An `nn_module` containing 163 parameters.
#> 
#> ── Modules ─────────────────────────────────────────────────────────────────────
#> • first: <nn_linear> #100 parameters
#> • second: <nn_linear> #63 parameters
```
