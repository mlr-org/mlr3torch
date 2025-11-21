# Learner Torch Model

Create a torch learner from an instantiated
[`nn_module()`](https://torch.mlverse.org/docs/reference/nn_module.html).
For classification, the output of the network must be the scores (before
the softmax).

## Parameters

See
[`LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)

## See also

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/reference/mlr_learners.mlp.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/reference/mlr_learners.module.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_image.md)

Other Graph Network:
[`ModelDescriptor()`](https://mlr3torch.mlr-org.com/reference/ModelDescriptor.md),
[`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md),
[`mlr_pipeops_module`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_module.md),
[`mlr_pipeops_torch`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch.md),
[`mlr_pipeops_torch_ingress`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress.md),
[`mlr_pipeops_torch_ingress_categ`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_categ.md),
[`mlr_pipeops_torch_ingress_ltnsr`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_ltnsr.md),
[`mlr_pipeops_torch_ingress_num`](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_torch_ingress_num.md),
[`model_descriptor_to_learner()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_learner.md),
[`model_descriptor_to_module()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_to_module.md),
[`model_descriptor_union()`](https://mlr3torch.mlr-org.com/reference/model_descriptor_union.md),
[`nn_graph()`](https://mlr3torch.mlr-org.com/reference/nn_graph.md)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3torch::LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)
-\> `LearnerTorchModel`

## Active bindings

- `ingress_tokens`:

  (named [`list()`](https://rdrr.io/r/base/list.html) with
  `TorchIngressToken` or `NULL`)  
  The ingress tokens. Must be non-`NULL` when calling `$train()`.

## Methods

### Public methods

- [`LearnerTorchModel$new()`](#method-LearnerTorchModel-new)

- [`LearnerTorchModel$clone()`](#method-LearnerTorchModel-clone)

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

    LearnerTorchModel$new(
      network = NULL,
      ingress_tokens = NULL,
      task_type,
      properties = NULL,
      optimizer = NULL,
      loss = NULL,
      callbacks = list(),
      packages = character(0),
      feature_types = NULL
    )

#### Arguments

- `network`:

  ([`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  An instantiated
  [`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html).
  Is not cloned during construction. For classification, outputs must be
  the scores (before the softmax).

- `ingress_tokens`:

  (`list` of
  [`TorchIngressToken()`](https://mlr3torch.mlr-org.com/reference/TorchIngressToken.md))  
  A list with ingress tokens that defines how the dataloader will be
  defined.

- `task_type`:

  (`character(1)`)  
  The task type.

- `properties`:

  (`NULL` or [`character()`](https://rdrr.io/r/base/character.html))  
  The properties of the learner. Defaults to all available properties
  for the given task type.

- `optimizer`:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md))  
  The torch optimizer.

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md))  
  The loss to use for training.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)s)  
  The callbacks used during training. Must have unique ids. They are
  executed in the order in which they are provided

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `feature_types`:

  (`NULL` or [`character()`](https://rdrr.io/r/base/character.html))  
  The feature types. Defaults to all available feature types.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchModel$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# We show the learner using a classification task

# The iris task has 4 features and 3 classes
network = nn_linear(4, 3)
task = tsk("iris")

# This defines the dataloader.
# It loads all 4 features, which are also numeric.
# The shape is (NA, 4) because the batch dimension is generally NA
ingress_tokens = list(
  input = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, 4))
)

# Creating the learner and setting required parameters
learner = lrn("classif.torch_model",
  network = network,
  ingress_tokens = ingress_tokens,
  batch_size = 16,
  epochs = 1,
  device = "cpu"
)

# A simple train-predict
ids = partition(task)
learner$train(task, ids$train)
learner$predict(task, ids$test)
#> 
#> ── <PredictionClassif> for 50 observations: ────────────────────────────────────
#>  row_ids     truth  response
#>        2    setosa virginica
#>        4    setosa virginica
#>        8    setosa virginica
#>      ---       ---       ---
#>      144 virginica virginica
#>      146 virginica virginica
#>      148 virginica virginica
```
