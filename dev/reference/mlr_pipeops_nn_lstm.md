# Long Short-Term Memory

For each element in the input sequence, each layer computes the
following function:

## nn_module

Calls
[`torch::nn_lstm()`](https://torch.mlverse.org/docs/reference/nn_lstm.html)
when trained, where the parameter `input_size` is inferred as the last
dimension of the input tensor and `batch_first` is always `TRUE`, see
section *Tensor Layout*.

## Parameters

The parameters are those of
[`nn("rnn")`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_rnn.md)
without `nonlinearity`, whose activation functions an LSTM cell fixes.

## Input and Output Channels

There is one input channel `"input"`, the sequence to run over.

The module has three outputs: `"output"`, the output sequence of shape
`(batch, sequence, hidden_size * directions)`, and `"h_n"` and `"c_n"`,
the final hidden state and the final cell state, both of shape
`(batch, num_layers * directions, hidden_size)`. Only `"output"` is an
output channel by default. Set `$outputs` to also (or only) get the
final states, e.g. `outputs = c("output", "h_n", "c_n")`.

For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## Tensor Layout

Input and output are `(batch, sequence, feature)`, i.e. the
`batch_first` layout of
[`torch::nn_rnn()`](https://torch.mlverse.org/docs/reference/nn_rnn.html),
which is fixed and not a hyperparameter. `torch` defaults to
`(sequence, batch, feature)`, but the first dimension of every shape has
to be the batch dimension here.

The final hidden state is `(layers * directions, batch, hidden_size)` in
`torch`, even in the batch-first layout. It is transposed to
`(batch, layers * directions, hidden_size)` here, so that the batch
dimension comes first for it as well.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\>
[`PipeOpTorchRecurrent`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_nn_recurrent.md)
-\> `PipeOpTorchLSTM`

## Methods

### Public methods

- [`PipeOpTorchLSTM$new()`](#method-PipeOpTorchLSTM-initialize)

- [`PipeOpTorchLSTM$clone()`](#method-PipeOpTorchLSTM-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchLSTM$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchLSTM$new(id = "nn_lstm", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchLSTM$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchLSTM$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("lstm", hidden_size = 10)
pipeop
#> 
#> ── PipeOp <lstm>: not trained ──────────────────────────────────────────────────
#> Values: hidden_size=10
#> 
#> ── Input channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>   input ModelDescriptor    Task
#> 
#> ── Output channels: 
#>    name           train predict
#>  <char>          <char>  <char>
#>  output ModelDescriptor    Task
# The available parameters
pipeop$param_set
#> <ParamSet(5)>
#>               id    class lower upper nlevels        default  value
#>           <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:   hidden_size ParamInt     1   Inf     Inf <NoDefault[0]>     10
#> 2:    num_layers ParamInt     1   Inf     Inf              1 [NULL]
#> 3:          bias ParamLgl    NA    NA       2           TRUE [NULL]
#> 4:       dropout ParamDbl     0     1     Inf              0 [NULL]
#> 5: bidirectional ParamLgl    NA    NA       2          FALSE [NULL]
```
