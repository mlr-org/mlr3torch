# Image Learner

Base Class for Image Learners. The features are assumed to be a single
[`lazy_tensor`](https://mlr3torch.mlr-org.com/reference/lazy_tensor.md)
column in RGB format.

## Parameters

Parameters include those inherited from
[`LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)
and the `param_set` construction argument.

## See also

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/reference/mlr_learners.mlp.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/reference/mlr_learners.module.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_model.md)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3torch::LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)
-\> `LearnerTorchImage`

## Methods

### Public methods

- [`LearnerTorchImage$new()`](#method-LearnerTorchImage-new)

- [`LearnerTorchImage$clone()`](#method-LearnerTorchImage-clone)

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

    LearnerTorchImage$new(
      id,
      task_type,
      param_set = ps(),
      label,
      optimizer = NULL,
      loss = NULL,
      callbacks = list(),
      packages,
      man,
      properties = NULL,
      predict_types = NULL,
      jittable = FALSE
    )

#### Arguments

- `id`:

  (`character(1)`)  
  The id for of the new object.

- `task_type`:

  (`character(1)`)  
  The task type.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  The parameter set.

- `label`:

  (`character(1)`)  
  Label for the new instance.

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

- `man`:

  (`character(1)`)  
  String in the format `[pkg]::[topic]` pointing to a manual page for
  this object. The referenced help package can be opened via method
  `$help()`.

- `properties`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The properties of the object. See
  [`mlr_reflections$learner_properties`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
  for available values.

- `predict_types`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The predict types. See
  [`mlr_reflections$learner_predict_types`](https://mlr3.mlr-org.com/reference/mlr_reflections.html)
  for available values.

- `jittable`:

  (`logical(1)`)  
  Whether the model can be jit-traced.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchImage$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
