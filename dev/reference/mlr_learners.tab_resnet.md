# Tabular ResNet

Tabular resnet.

## Dictionary

This [Learner](https://mlr3.mlr-org.com/reference/Learner.html) can be
instantiated using the sugar function
[`lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html):

    lrn("classif.tab_resnet", ...)
    lrn("regr.tab_resnet", ...)

## Properties

- Supported task types: 'classif', 'regr'

- Predict Types:

  - classif: 'response', 'prob'

  - regr: 'response'

- Feature Types: “integer”, “numeric”, “lazy_tensor”

- Required Packages: [mlr3](https://CRAN.R-project.org/package=mlr3),
  [mlr3torch](https://CRAN.R-project.org/package=mlr3torch),
  [torch](https://CRAN.R-project.org/package=torch)

## Parameters

Parameters from
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
as well as:

- `n_blocks` :: `integer(1)`  
  The number of blocks.

- `d_block` :: `integer(1)`  
  The input and output dimension of a block.

- `d_hidden` :: `integer(1)`  
  The latent dimension of a block.

- `d_hidden_multiplier` :: `numeric(1)`  
  Alternative way to specify the latent dimension as
  `d_block * d_hidden_multiplier`.

- `dropout1` :: `numeric(1)`  
  First dropout ratio.

- `dropout2` :: `numeric(1)`  
  Second dropout ratio.

- `shape` :: [`integer()`](https://rdrr.io/r/base/integer.html) or
  `NULL`  
  Shape of the input tensor. Only needs to be provided if the input is a
  lazy tensor with unknown shape.

## References

Gorishniy Y, Rubachev I, Khrulkov V, Babenko A (2021). “Revisiting Deep
Learning for Tabular Data.” *arXiv*, **2106.11959**.

## See also

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.mlp.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.module.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3torch::LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
-\> `LearnerTorchTabResNet`

## Methods

### Public methods

- [`LearnerTorchTabResNet$new()`](#method-LearnerTorchTabResNet-new)

- [`LearnerTorchTabResNet$clone()`](#method-LearnerTorchTabResNet-clone)

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
- [`mlr3torch::LearnerTorch$dataset()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-dataset)
- [`mlr3torch::LearnerTorch$format()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-format)
- [`mlr3torch::LearnerTorch$marshal()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-marshal)
- [`mlr3torch::LearnerTorch$print()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-print)
- [`mlr3torch::LearnerTorch$unmarshal()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-unmarshal)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    LearnerTorchTabResNet$new(
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

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchTabResNet$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Define the Learner and set parameter values
learner = lrn("classif.tab_resnet")
learner$param_set$set_values(
  epochs = 1, batch_size = 16, device = "cpu",
  n_blocks = 2, d_block = 10, d_hidden = 20, dropout1 = 0.3, dropout2 = 0.3
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
#>       0.68 
```
