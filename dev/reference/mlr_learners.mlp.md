# Multi Layer Perceptron

Fully connected feed forward network with dropout after each activation
function. The features can either be a single
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
or one or more numeric columns (but not both).

## Dictionary

This [Learner](https://mlr3.mlr-org.com/reference/Learner.html) can be
instantiated using the sugar function
[`lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html):

    lrn("classif.mlp", ...)
    lrn("regr.mlp", ...)

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

- `activation` :: `[nn_module]`  
  The activation function. Is initialized to
  [`nn_relu`](https://torch.mlverse.org/docs/reference/nn_relu.html).

- `activation_args` :: named
  [`list()`](https://rdrr.io/r/base/list.html)  
  A named list with initialization arguments for the activation
  function. This is intialized to an empty list.

- `neurons` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The number of neurons per hidden layer. By default there is no hidden
  layer. Setting this to `c(10, 20)` would have a the first hidden layer
  with 10 neurons and the second with 20.

- `n_layers` :: [`integer()`](https://rdrr.io/r/base/integer.html)  
  The number of layers. This parameter must only be set when `neurons`
  has length 1.

- `p` :: `numeric(1)`  
  The dropout probability. Is initialized to `0.5`.

- `shape` :: [`integer()`](https://rdrr.io/r/base/integer.html) or
  `NULL`  
  The input shape of length 2, e.g. `c(NA, 5)`. Only needs to be present
  when there is a lazy tensor input with unknown shape (`NULL`).
  Otherwise the input shape is inferred from the number of numeric
  features.

## References

Gorishniy Y, Rubachev I, Khrulkov V, Babenko A (2021). “Revisiting Deep
Learning for Tabular Data.” *arXiv*, **2106.11959**.

## See also

Other Learner:
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.module.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3torch::LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
-\> `LearnerTorchMLP`

## Methods

### Public methods

- [`LearnerTorchMLP$new()`](#method-LearnerTorchMLP-new)

- [`LearnerTorchMLP$clone()`](#method-LearnerTorchMLP-clone)

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

    LearnerTorchMLP$new(
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

    LearnerTorchMLP$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Define the Learner and set parameter values
learner = lrn("classif.mlp")
learner$param_set$set_values(
  epochs = 1, batch_size = 16, device = "cpu",
  neurons = 10
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
#>       0.66 
```
