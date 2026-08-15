# Categorical Tokenizer

Tokenizes categorical features into a dense embedding. For an input of
shape `(batch, n_features)` the output shape is
`(batch, n_features, d_token)`.

## nn_module

Calls
[`nn_tokenizer_categ()`](https://mlr3torch.mlr-org.com/dev/reference/nn_tokenizer_categ.md)
when trained where the parameter `cardinalities` is inferred. The output
shape is `(batch, n_features, d_token)`.

## Parameters

- `d_token` :: `integer(1)`  
  The dimension of the embedding.

- `bias` :: `logical(1)`  
  Whether to use a bias. Is initialized to `TRUE`.

- `initialization` :: `character(1)`  
  The initialization method for the embedding weights. Possible values
  are `"uniform"` (default) and `"normal"`.

- `cardinalities` ::
  [`integer()`](https://rdrr.io/r/base/integer.html)  
  The number of categories for each feature. Only needs to be provided
  when working with
  [`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
  inputs.

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
-\> `PipeOpTorchTokenizerCateg`

## Methods

### Public methods

- [`PipeOpTorchTokenizerCateg$new()`](#method-PipeOpTorchTokenizerCateg-initialize)

- [`PipeOpTorchTokenizerCateg$clone()`](#method-PipeOpTorchTokenizerCateg-clone)

Inherited methods

- [`mlr3pipelines::PipeOp$help()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-help)
- [`mlr3pipelines::PipeOp$predict()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-predict)
- [`mlr3pipelines::PipeOp$print()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-print)
- [`mlr3pipelines::PipeOp$train()`](https://mlr3pipelines.mlr-org.com/reference/PipeOp.html#method-train)
- [`PipeOpTorch$shapes_out()`](https://mlr3torch.mlr-org.com/dev/reference/PipeOpTorch.html#method-shapes_out)

------------------------------------------------------------------------

### `PipeOpTorchTokenizerCateg$new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    PipeOpTorchTokenizerCateg$new(id = "nn_tokenizer_categ", param_vals = list())

#### Arguments

- `id`:

  (`character(1)`)  
  Identifier of the resulting object.

- `param_vals`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  List of hyperparameter settings, overwriting the hyperparameter
  settings that would otherwise be set during construction.

------------------------------------------------------------------------

### `PipeOpTorchTokenizerCateg$clone()`

The objects of this class are cloneable with this method.

#### Usage

    PipeOpTorchTokenizerCateg$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
# Construct the PipeOp
pipeop = nn("tokenizer_categ", d_token = 10)
pipeop
#> 
#> ── PipeOp <tokenizer_categ>: not trained ───────────────────────────────────────
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
#> <ParamSet(4)>
#>                id    class lower upper nlevels        default   value
#>            <char>   <char> <num> <num>   <num>         <list>  <list>
#> 1:        d_token ParamInt     1   Inf     Inf <NoDefault[0]>      10
#> 2:           bias ParamLgl    NA    NA       2 <NoDefault[0]>    TRUE
#> 3: initialization ParamFct    NA    NA       2 <NoDefault[0]> uniform
#> 4:  cardinalities ParamUty    NA    NA     Inf <NoDefault[0]>  [NULL]
```
