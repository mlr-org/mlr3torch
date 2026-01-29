# History Callback

Saves the training and validation history during training. The history
is saved as a data.table where the validation measures are prefixed with
`"valid."` and the training measures are prefixed with `"train."`.

## Super class

[`mlr3torch::CallbackSet`](https://mlr3torch.mlr-org.com/dev/reference/mlr_callback_set.md)
-\> `CallbackSetHistory`

## Methods

### Public methods

- [`CallbackSetHistory$on_begin()`](#method-CallbackSetHistory-on_begin)

- [`CallbackSetHistory$state_dict()`](#method-CallbackSetHistory-state_dict)

- [`CallbackSetHistory$load_state_dict()`](#method-CallbackSetHistory-load_state_dict)

- [`CallbackSetHistory$on_before_valid()`](#method-CallbackSetHistory-on_before_valid)

- [`CallbackSetHistory$on_epoch_end()`](#method-CallbackSetHistory-on_epoch_end)

- [`CallbackSetHistory$clone()`](#method-CallbackSetHistory-clone)

Inherited methods

- [`mlr3torch::CallbackSet$print()`](https://mlr3torch.mlr-org.com/dev/reference/CallbackSet.html#method-print)

------------------------------------------------------------------------

### Method `on_begin()`

Initializes lists where the train and validation metrics are stored.

#### Usage

    CallbackSetHistory$on_begin()

------------------------------------------------------------------------

### Method `state_dict()`

Converts the lists to data.tables.

#### Usage

    CallbackSetHistory$state_dict()

------------------------------------------------------------------------

### Method [`load_state_dict()`](https://torch.mlverse.org/docs/reference/load_state_dict.html)

Sets the field `$train` and `$valid` to those contained in the state
dict.

#### Usage

    CallbackSetHistory$load_state_dict(state_dict)

#### Arguments

- `state_dict`:

  (`callback_state_history`)  
  The state dict as retrieved via `$state_dict()`.

------------------------------------------------------------------------

### Method `on_before_valid()`

Add the latest training scores to the history.

#### Usage

    CallbackSetHistory$on_before_valid()

------------------------------------------------------------------------

### Method `on_epoch_end()`

Add the latest validation scores to the history.

#### Usage

    CallbackSetHistory$on_epoch_end()

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    CallbackSetHistory$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.

## Examples

``` r
cb = t_clbk("history")
task = tsk("iris")

learner = lrn("classif.mlp", epochs = 3, batch_size = 1,
  callbacks = t_clbk("history"), validate = 0.3)
learner$param_set$set_values(
  measures_train = msrs(c("classif.acc", "classif.ce")),
  measures_valid = msr("classif.ce")
)
learner$train(task)

print(learner$model$callbacks$history)
#> Key: <epoch>
#>    epoch train.classif.acc train.classif.ce valid.classif.ce
#>    <num>             <num>            <num>            <num>
#> 1:     1         0.6666667        0.3333333        0.3333333
#> 2:     2         0.6571429        0.3428571        0.3555556
#> 3:     3         0.5904762        0.4095238        0.2888889
```
