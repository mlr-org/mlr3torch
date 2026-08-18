# Encode the Network Output as a Prediction

Converts the raw output of a network into a
[`list()`](https://rdrr.io/r/base/list.html) that can be passed to
[`mlr3::as_prediction_data()`](https://mlr3.mlr-org.com/reference/as_prediction_data.html),
which is what the private `.encode_prediction()` method of a
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
has to return.

This is the default implementation that is used by
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
and
[`LearnerTorchModel`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_model.md),
i.e. by all learners that don't overwrite `.encode_prediction()`. When
adding support for a custom task type, implement a method for the
corresponding [`Task`](https://mlr3.mlr-org.com/reference/Task.html)
class, which makes the generic torch learners work for that task type.

For the network output that is expected for the built-in task types, see
section *Network Head and Target Encoding* of
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md).

## Usage

``` r
encode_prediction(task, network_output, predict_type, ...)
```

## Arguments

- task:

  ([`Task`](https://mlr3.mlr-org.com/reference/Task.html))  
  The task to predict on.

- network_output:

  ([`torch_tensor`](https://torch.mlverse.org/docs/reference/torch_tensor.html)
  or [`list()`](https://rdrr.io/r/base/list.html) of them)  
  The raw output of the network in evaluation mode. A network with more
  than one head – e.g. one predicting a mean and a standard deviation –
  returns a [`list()`](https://rdrr.io/r/base/list.html) of tensors,
  which is passed on unchanged. The encodings of the built-in task types
  expect a single tensor.

- predict_type:

  (`character(1)`)  
  The predict type of the learner, e.g. `"response"` or `"prob"`.

- ...:

  (any)  
  Additional arguments. Not used yet.

## Value

named [`list()`](https://rdrr.io/r/base/list.html)
