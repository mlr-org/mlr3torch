# Graph Learner for Torch Networks

The
[`GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html)
that
[`as_learner_torch()`](https://mlr3torch.mlr-org.com/dev/reference/as_learner_torch.md)
returns. On top of a `GraphLearner` it has the `$network`, `$loss`,
`$optimizer` and `$callbacks` fields and the `$dataset()` method of a
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).
It is not in the
[`mlr_learners`](https://mlr3.mlr-org.com/reference/mlr_learners.html)
dictionary, because it cannot be constructed without a graph.

## See also

Other Learner:
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
[`mlr3pipelines::GraphLearner`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html)
-\> `GraphLearnerTorch`

## Active bindings

- `network`:

  ([`nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html))  
  The network of the trained learner, i.e. `$base_learner()$network`.

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md))  
  The torch loss, i.e. the one of the graph's
  [`PipeOpTorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_loss.md).
  Read-only, the graph configures it.

- `optimizer`:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md))  
  The torch optimizer, i.e. the one of the graph's
  [`PipeOpTorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_optimizer.md).
  Read-only, the graph configures it.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md)s)  
  The callbacks, i.e. those of the graph's
  [`PipeOpTorchCallbacks`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch_callbacks.md).
  Read-only, the graph configures them.

## Methods

### Public methods

- [`GraphLearnerTorch$dataset()`](#method-GraphLearnerTorch-dataset)

- [`GraphLearnerTorch$clone()`](#method-GraphLearnerTorch-clone)

Inherited methods

- [`mlr3::Learner$configure()`](https://mlr3.mlr-org.com/reference/Learner.html#method-configure)
- [`mlr3::Learner$encapsulate()`](https://mlr3.mlr-org.com/reference/Learner.html#method-encapsulate)
- [`mlr3::Learner$format()`](https://mlr3.mlr-org.com/reference/Learner.html#method-format)
- [`mlr3::Learner$help()`](https://mlr3.mlr-org.com/reference/Learner.html#method-help)
- [`mlr3::Learner$predict()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict)
- [`mlr3::Learner$predict_newdata()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict_newdata)
- [`mlr3::Learner$reset()`](https://mlr3.mlr-org.com/reference/Learner.html#method-reset)
- [`mlr3::Learner$train()`](https://mlr3.mlr-org.com/reference/Learner.html#method-train)
- [`mlr3pipelines::GraphLearner$base_learner()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-base_learner)
- [`mlr3pipelines::GraphLearner$ids()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-ids)
- [`mlr3pipelines::GraphLearner$importance()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-importance)
- [`mlr3pipelines::GraphLearner$initialize()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-initialize)
- [`mlr3pipelines::GraphLearner$loglik()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-loglik)
- [`mlr3pipelines::GraphLearner$marshal()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-marshal)
- [`mlr3pipelines::GraphLearner$oob_error()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-oob_error)
- [`mlr3pipelines::GraphLearner$plot()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-plot)
- [`mlr3pipelines::GraphLearner$predict_newdata_fast()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-predict_newdata_fast)
- [`mlr3pipelines::GraphLearner$print()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-print)
- [`mlr3pipelines::GraphLearner$selected_features()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-selected_features)
- [`mlr3pipelines::GraphLearner$unmarshal()`](https://mlr3pipelines.mlr-org.com/reference/mlr_learners_graph.html#method-unmarshal)

------------------------------------------------------------------------

### `GraphLearnerTorch$dataset()`

Create the dataset for a task, i.e. the tensors that are fed to the
network.

#### Usage

    GraphLearnerTorch$dataset(
      task,
      stage = if (is.null(self$model)) "train" else "predict"
    )

#### Arguments

- `task`:

  [`Task`](https://mlr3.mlr-org.com/reference/Task.html)  
  The task.

- `stage`:

  (`character(1)`)  
  Whether to create the dataset the way `$train()` does (`"train"`) or
  the way `$predict()` does (`"predict"`). Defaults to `"predict"` for a
  trained learner and to `"train"` otherwise, because the prediction
  phase reuses the state that the operators before the ingress fitted
  during training.

#### Returns

[`dataset`](https://torch.mlverse.org/docs/reference/dataset.html)

------------------------------------------------------------------------

### `GraphLearnerTorch$clone()`

The objects of this class are cloneable with this method.

#### Usage

    GraphLearnerTorch$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
