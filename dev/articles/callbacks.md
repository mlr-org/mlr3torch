# Custom Callbacks

The torch callback mechanism allows to customize the training loop of a
neural network. While `mlr3torch` has some predefined callbacks for
common use-cases, this vignette will show you how to write your own
custom callback.

### Building Blocks

At the heart of the callback mechanism are three `R6` classes:

- `CallbackSet` contains the methods that are executed at different
  stages of the training loop
- `TorchCallback` wraps a `CallbackSet` and annotates it with meta
  information, including a `ParamSet`
- `ContextTorch` defines which information of the training process the
  `CallbackSet` has access to

When using predefined callbacks, one usually only interacts with the
`TorchCallback` class, which is constructed using `t_clbk(<id>)`:

``` r
tc_hist = t_clbk("history")
tc_hist
```

    ## <TorchCallback:history> History
    ## * Generator: CallbackSetHistory
    ## * Parameters: list()
    ## * Packages: mlr3torch,torch

The wrapped `CallbackSet` is accessible via the field `$generator`, and
we can create a new instance by calling `$new()`

``` r
cbs_hist = tc_hist$generator$new()
```

Within the training loop of a torch model, different stages exist at
which a callback can execute code. Below, we list all stages that are
available, but they are also described in the documentation of
`CallbackSet`.

``` r
mlr_reflections$torch$callback_stages
```

    ##  [1] "on_begin"             "on_epoch_begin"       "on_batch_begin"      
    ##  [4] "on_after_backward"    "on_batch_end"         "on_before_valid"     
    ##  [7] "on_batch_valid_begin" "on_batch_valid_end"   "on_valid_end"        
    ## [10] "on_epoch_end"         "on_end"               "on_exit"

For the history callback, we see that it runs code at the beginning,
before validation starts, and at the end of an epoch.

``` r
cbs_hist
```

    ## <CallbackSetHistory>
    ## * Stages: on_begin, on_before_valid, on_epoch_end

### Writing a Custom Logger

In order to define our own custom callback, we can use the helper
[`torch_callback()`](https://mlr3torch.mlr-org.com/dev/reference/torch_callback.md)
helper function. As an example, we will create a custom logger that
keeps track of an exponential moving average of the train loss and
prints it at the end of every epoch. This callback takes one argument
`alpha` which is the smoothing parameter and will store the moving
average in `self$moving_loss`. The value `alpha` can later be configured
in the `Learner`.

Then, we implement the main logic of the callback using two stages:

- `on_batch_end()`, which is called after the optimizer updates the
  network parameters using a mini-batch. Here, we access the last loss
  via the `ContextTorch`’s (accessible via `self$ctx`) `$last_loss`
  field and update the value `self$moving_loss`.
- `on_before_valid()`, which is run before the validation loop. At this
  point we simply print the exponential moving average of the training
  loss.

Finally, in order to make the final value of the moving average
accessible in the `Learner` after the training finishes, we implement
the

- `state_dict()` method to return this value and the
- `load_state_dict(state_dict)` method, that takes a previously
  retrieved state dict and sets it in the callback.

Note that we only do this here to have access to the final
`$moving_loss` via the learner, but we would otherwise not have to
implement these methods.

``` r
custom_logger = torch_callback("custom_logger",
  initialize = function(alpha = 0.1) {
    self$alpha = alpha
    self$moving_loss = NULL
  },
  on_batch_end = function() {
    if (is.null(self$moving_training_loss)) {
      self$moving_loss = self$ctx$last_loss
    } else {
      self$moving_loss = self$alpha * self$last_loss + (1 - self$alpha) * self$moving_loss
    }
  },
  on_before_valid = function() {
    cat(sprintf("Epoch %s: %.2f\n", self$ctx$epoch, self$moving_loss))
  },
  state_dict = function() {
    self$moving_loss
  },
  load_state_dict = function(state_dict) {
    self$moving_loss = state_dict
  }
)
```

This created a `TorchCallback` object and and associated `CallbackSet`
R6-class for us:

``` r
custom_logger
```

    ## <TorchCallback:custom_logger> Custom_logger
    ## * Generator: CallbackSetCustom_logger
    ## * Parameters: list()
    ## * Packages: mlr3torch,torch

``` r
custom_logger$generator
```

    ## <CallbackSetCustom_logger> object generator
    ##   Inherits from: <inherit>
    ##   Public:
    ##     initialize: function (alpha = 0.1) 
    ##     state_dict: function () 
    ##     load_state_dict: function (state_dict) 
    ##     on_before_valid: function () 
    ##     on_batch_end: function () 
    ##   Parent env: <environment: 0x56534e1c33e8>
    ##   Locked objects: FALSE
    ##   Locked class: FALSE
    ##   Portable: TRUE

### Using the Custom Logger

In order to showcase its effects, we create train a simple multi layer
perceptron and pass it the callback. We set the smoothing parameter
`alpha` to 0.05.

``` r
task = tsk("iris")
mlp = lrn("classif.mlp",
  callbacks = custom_logger, cb.custom_logger.alpha = 0.05,
  epochs = 5, batch_size = 64, neurons = 20)

mlp$train(task)
```

    ## Epoch 1: 1.22
    ## Epoch 2: 0.93
    ## Epoch 3: 1.19
    ## Epoch 4: 1.39
    ## Epoch 5: 1.05

The information that is returnede by `state_dict()` is now accessible
via the `Learner`’s `$model`-slot:

``` r
mlp$model$callbacks$custom_logger
```

    ## [1] 1.047876
