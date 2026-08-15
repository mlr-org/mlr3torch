# Numeric Tokenizer

Tokenizes numeric features into a dense embedding. For an input of shape
`(batch, n_features)` the output shape is
`(batch, n_features, d_token)`.

## nn_module

Calls
[`nn_tokenizer_num()`](https://mlr3torch.mlr-org.com/dev/reference/nn_tokenizer_num.md)
when trained where the parameter `n_features` is inferred. The output
shape is `(batch, n_features, d_token)`.

## Parameters

- `d_token` :: `integer(1)`  
  The dimension of the embedding.

- `bias` :: `logical(1)`  
  Whether to use a bias. Is initialized to `TRUE`.

- `initialization` :: `character(1)`  
  The initialization method for the embedding weights. Possible values
  are `"uniform"` (default) and `"normal"`.

## Input and Output Channels

One input channel called `"input"` and one output channel called
`"output"`. For an explanation see
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md).

## State

The state is the value calculated by the public method `$shapes_out()`.

## Super classes

[`mlr3pipelines::PipeOp`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html)
-\>
[`PipeOpTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_torch.md)
-\> `PipeOpTorchTokenizerNum`

## Methods

### Public methods

- [`PipeOpTorchTokenizerNum$new()`](#method-PipeOpTorchTokenizerNum-initialize)

- [`PipeOpTorchTokenizerNum$clone()`](#method-PipeOpTorchTokenizerNum-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchTokenizerNum$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchTokenizerNum$new(id = "nn_tokenizer_num", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchTokenizerNum$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchTokenizerNum$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("tokenizer_num", d_token = 10)
pipeop
#> 
#> ── PipeOp <tokenizer_num>: not trained ─────────────────────────────────────────
#> Values: d_token=10, bias=TRUE, initialization=uniform
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
#> <ParamSet(3)>
#>                id    class lower upper nlevels        default   value
#>            <char>   <char> <num> <num>   <num>         <list>  <list>
#> 1:        d_token ParamInt     1   Inf     Inf <NoDefault[0]>      10
#> 2:           bias ParamLgl    NA    NA       2 <NoDefault[0]>    TRUE
#> 3: initialization ParamFct    NA    NA       2 <NoDefault[0]> uniform
```
