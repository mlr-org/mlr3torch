# Base Class for Torch Descriptors

Abstract Base Class from which
[`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md),
and
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)
inherit. This class wraps a generator (R6Class Generator or the torch
version of such a generator) and annotates it with metadata such as a
[`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html), a
label, an ID, packages, or a manual page.

The parameters are the construction arguments of the wrapped generator
and the parameter `$values` are passed to the generator when calling the
public method `$generate()`.

## Parameters

Defined by the constructor argument `param_set`. All parameters are
tagged with `"train"`, but this is done automatically during initialize.

## See also

Other Torch Descriptor:
[`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md),
[`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md),
[`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md),
[`as_torch_callbacks()`](https://mlr3torch.mlr-org.com/reference/as_torch_callbacks.md),
[`as_torch_loss()`](https://mlr3torch.mlr-org.com/reference/as_torch_loss.md),
[`as_torch_optimizer()`](https://mlr3torch.mlr-org.com/reference/as_torch_optimizer.md),
[`mlr3torch_losses`](https://mlr3torch.mlr-org.com/reference/mlr3torch_losses.md),
[`mlr3torch_optimizers`](https://mlr3torch.mlr-org.com/reference/mlr3torch_optimizers.md),
[`t_clbk()`](https://mlr3torch.mlr-org.com/reference/t_clbk.md),
[`t_loss()`](https://mlr3torch.mlr-org.com/reference/t_loss.md),
[`t_opt()`](https://mlr3torch.mlr-org.com/reference/t_opt.md)

## Public fields

- `label`:

  (`character(1)`)  
  Label for this object. Can be used in tables, plot and text output
  instead of the ID.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  Set of hyperparameters.

- `packages`:

  (`character(1)`)  
  Set of required packages. These packages are loaded, but not attached.

- `id`:

  (`character(1)`)  
  Identifier of the object. Used in tables, plot and text output.

- `generator`:

  The wrapped generator that is described.

- `man`:

  (`character(1)`)  
  String in the format `[pkg]::[topic]` pointing to a manual page for
  this object.

## Active bindings

- `phash`:

  (`character(1)`)  
  Hash (unique identifier) for this partial object, excluding some
  components which are varied systematically (e.g. the parameter
  values).

## Methods

### Public methods

- [`TorchDescriptor$new()`](#method-TorchDescriptor-new)

- [`TorchDescriptor$print()`](#method-TorchDescriptor-print)

- [`TorchDescriptor$generate()`](#method-TorchDescriptor-generate)

- [`TorchDescriptor$help()`](#method-TorchDescriptor-help)

- [`TorchDescriptor$clone()`](#method-TorchDescriptor-clone)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    TorchDescriptor$new(
      generator,
      id = NULL,
      param_set = NULL,
      packages = NULL,
      label = NULL,
      man = NULL,
      additional_args = NULL
    )

#### Arguments

- `generator`:

  The wrapped generator that is described.

- `id`:

  (`character(1)`)  
  The id for of the new object.

- `param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html))  
  The parameter set.

- `packages`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The R packages this object depends on.

- `label`:

  (`character(1)`)  
  Label for the new instance.

- `man`:

  (`character(1)`)  
  String in the format `[pkg]::[topic]` pointing to a manual page for
  this object. The referenced help package can be opened via method
  `$help()`.

- `additional_args`:

  ([`list()`](https://rdrr.io/r/base/list.html))  
  Additional arguments if necessary. For learning rate schedulers, this
  is the torch::LRScheduler.

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Prints the object

#### Usage

    TorchDescriptor$print(...)

#### Arguments

- `...`:

  any

------------------------------------------------------------------------

### Method `generate()`

Calls the generator with the given parameter values.

#### Usage

    TorchDescriptor$generate()

------------------------------------------------------------------------

### Method [`help()`](https://rdrr.io/r/utils/help.html)

Displays the help file of the wrapped object.

#### Usage

    TorchDescriptor$help()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    TorchDescriptor$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
