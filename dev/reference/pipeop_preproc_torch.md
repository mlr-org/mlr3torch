# Create Torch Preprocessing PipeOps

Function to create objects of class
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md)
in a more convenient way. Start by reading the documentation of
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md).

## Usage

``` r
pipeop_preproc_torch(
  id,
  fn,
  shapes_out = NULL,
  param_set = NULL,
  packages = character(0),
  rowwise = FALSE,
  parent_env = parent.frame(),
  stages_init = NULL,
  tags = NULL
)
```

## Arguments

- id:

  (`character(1)`)  
  The id for of the new object.

- fn:

  (`function`)  
  The preprocessing function.

- shapes_out:

  (`function` or `NULL` or `"infer"`)  
  The private `.shapes_out(shapes_in, param_vals, task)` method of
  [`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md)
  (see section Inheriting). Special values are `NULL` and `"infer"`: If
  `NULL`, the output shapes are unknown. Option `"infer"` uses
  [`infer_shapes`](https://mlr3torch.mlr-org.com/dev/reference/infer_shapes.md).
  Method `"infer"` should be correct in most cases, but might fail in
  some edge cases.

- param_set:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) or
  `NULL`)  
  The parameter set. If this is left as `NULL` (default) the parameter
  set is inferred in the following way: All parameters but the first and
  `...` of `fn` are set as untyped parameters with tags 'train' and
  those that have no default value are tagged as 'required' as well.
  Default values are not annotated.

- packages:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- rowwise:

  (`logical(1)`)  
  Whether the preprocessing is applied row-wise.

- parent_env:

  (`environment`)  
  The parent environment for the R6 class.

- stages_init:

  (`character(1)`)  
  Initial value for the `stages` parameter. If `NULL` (default), will be
  set to `"both"` in case the `id` starts with `"trafo"` and to
  `"train"` if it starts with `"augment"`. Otherwise it must specified.

- tags:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Tags for the pipeop

## Value

An [`R6Class`](https://r6.r-lib.org/reference/R6Class.html) instance
inheriting from
[`PipeOpTaskPreprocTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_pipeops_preproc_torch.md)

## Examples

``` r
PipeOpPreprocExample = pipeop_preproc_torch("preproc_example", function(x, a) x + a)
po_example = PipeOpPreprocExample$new()
po_example$param_set
#> <ParamSet(3)>
#>                id    class lower upper nlevels        default  value
#>            <char>   <char> <num> <num>   <num>         <list> <list>
#> 1:              a ParamUty    NA    NA     Inf <NoDefault[0]> [NULL]
#> 2:         stages ParamFct    NA    NA       3 <NoDefault[0]> [NULL]
#> 3: affect_columns ParamUty    NA    NA     Inf  <Selector[1]> [NULL]
```
