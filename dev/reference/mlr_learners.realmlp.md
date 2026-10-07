# RealMLP

RealMLP is an MLP for tabular data with tuned defaults (RealMLP-TD),
introduced by Holzmüller (2024). The learner implements the
preprocessing, architecture and training recipe of the reference
implementation.

It is a special learner, as it implements it's own learning rate
scheduling per parameter group, as well as custom weight decay handling.
Configuring a custom learning reate scheduler as a callback or setting
the weight decay of an optimizer will therefore lead to unexpected
results.

## Dictionary

This [Learner](https://mlr3.mlr-org.com/reference/Learner.html) can be
instantiated using the sugar function
[`lrn()`](https://mlr3.mlr-org.com/reference/mlr_sugar.html):

    lrn("classif.realmlp", ...)
    lrn("regr.realmlp", ...)

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

Changed defaults of
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md):

- `epochs = 256`

- `batch_size = 256`

- `batch_size_predict = 1024`

- `drop_last = TRUE` and the optimizer is *adam* with

- `betas = c(0.9, 0.95)`

- `lr = 0.04` (classification) or `lr = 0.2` (regression).

Parameters from
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
as well as:

- `n_hidden_layers` :: `integer(1)`  
  The number of hidden layers. Default is `3`.

- `hidden_width` :: `integer(1)`  
  The width of the hidden layers. Default is `256`.

- `act` :: `character(1)`  
  The activation function, one of `"selu"`, `"mish"`, `"relu"`,
  `"silu"`, `"gelu"` and `"elu"`. Default is `"selu"` for classification
  and `"mish"` for regression.

- `use_parametric_act` :: `logical(1)`  
  Whether to use parametric activations \\x + \alpha (f(x) - x)\\ with a
  learned \\\alpha\\ per neuron, initialized with one. Default is
  `TRUE`.

- `p_drop` :: `numeric(1)`  
  The dropout probability. Default is `0.15`.

- `p_drop_sched` :: `character(1)`  
  The schedule of the dropout probability. Default is `"flat_cos"`.

- `wd` :: `numeric(1)`  
  The decoupled weight decay. Biases are not decayed. Default is `0.02`.

- `wd_sched` :: `character(1)`  
  The schedule of the weight decay. Default is `"flat_cos"`.

- `lr_sched` :: `character(1)`  
  The schedule of the learning rate. Default is `"coslog4"`.

- `bias_lr_factor`, `act_lr_factor`, `scale_lr_factor`, `plr_lr_factor`
  :: `numeric(1)`  
  The learning rate factors of the biases (default `0.1`), the
  parametric activations (default `0.1`), the scaling layer (default
  `6`) and the periodic embeddings (default `0.1`). All other parameters
  have the factor `1`.

- `add_front_scale` :: `logical(1)`  
  Whether the first layer starts with a learned elementwise scaling.
  Default is `TRUE`.

- `num_emb_type` :: `character(1)`  
  The embedding of the numerical features, one of:

  - `"pbld"` (default): \\\mathrm{concat}(W_2 \cos(2 \pi w_1 x + b_1) +
    b_2, x)\\.

  - `"pblrd"`: as `"pbld"`, with a ReLU after the second linear layer.

  - `"pl"`: \\W_2 \mathrm{concat}(\cos(2 \pi w_1 x), \sin(2 \pi w_1
    x)) + b_2\\.

  - `"plr"`: as `"pl"`, with a ReLU after the second linear layer.

  - `"none"`: no embedding.

- `plr_sigma` :: `numeric(1)`  
  The standard deviation of the initialization of the frequencies
  \\w_1\\. Default is `0.1`.

- `plr_hidden_1` :: `integer(1)`  
  The number of frequencies (`"pbld"`, `"pblrd"`) or twice the number of
  frequencies (`"pl"`, `"plr"`, must be even) per feature. Default is
  `16`.

- `plr_hidden_2` :: `integer(1)`  
  The embedding size per feature, including the appended input of
  `"pbld"` and `"pblrd"`. Default is `4`.

- `max_one_hot_cat_size` :: `integer(1)`  
  Categorical features with at most `max_one_hot_cat_size - 1` levels
  are one-hot encoded, all others are embedded. Default is `9`.

- `embedding_size` :: `integer(1)`  
  The embedding size of the categorical features. Default is `8`.

Only for classification:

- `ls_eps` :: `numeric(1)`  
  The label smoothing parameter. Requires the `cross_entropy` loss. The
  smoothing term ignores the loss's `class_weight` and `ignore_index`.
  Default is `0.1`.

Only for regression:

- `normalize_output` :: `logical(1)`  
  Whether to standardize the target during training. Default is `TRUE`.

- `clamp_output` :: `logical(1)`  
  Whether to clamp predictions to the range of the training target.
  Default is `TRUE`.

The schedules, as functions of the fraction \\t\\ of training steps
already taken, are:

- `"coslog4"`: \\(1 - \cos(2 \pi \log_2(1 + 15 t))) / 2\\, four cycles
  of increasing length.

- `"flat_cos"`: one for \\t \< 0.5\\, followed by a cosine decay to
  zero.

- `"cos"`: a cosine decay from one to zero.

- `"linear"`: \\1 - t\\.

- `"constant"`: one.

## References

Holzmüller D, Grinsztajn L, Steinwart I (2024). “Better by Default:
Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data.” In *Advances
in Neural Information Processing Systems (NeurIPS)*, volume 37.
2407.04491.

## See also

Other Learner:
[`GraphLearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/GraphLearnerTorch.md),
[`as_learner_torch()`](https://mlr3torch.mlr-org.com/dev/reference/as_learner_torch.md),
[`mlr_learners.ft_transformer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.ft_transformer.md),
[`mlr_learners.mlp`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.mlp.md),
[`mlr_learners.module`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.module.md),
[`mlr_learners.tab_resnet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tab_resnet.md),
[`mlr_learners.tabm`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.tabm.md),
[`mlr_learners.torch_featureless`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners.torch_featureless.md),
[`mlr_learners_torch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md),
[`mlr_learners_torch_image`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md),
[`mlr_learners_torch_model`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md)

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
-\> `LearnerTorchRealMLP`

## Methods

### Public methods

- [`LearnerTorchRealMLP$new()`](#method-LearnerTorchRealMLP-initialize)

- [`LearnerTorchRealMLP$clone()`](#method-LearnerTorchRealMLP-clone)

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

### `LearnerTorchRealMLP$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    LearnerTorchRealMLP$new(
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

### `LearnerTorchRealMLP$clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchRealMLP$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Define the Learner and set parameter values
learner = lrn("classif.realmlp")
learner$param_set$set_values(
  epochs = 1, batch_size = 16, device = "cpu",
  n_hidden_layers = 2, hidden_width = 32
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
#>       0.16 
```
