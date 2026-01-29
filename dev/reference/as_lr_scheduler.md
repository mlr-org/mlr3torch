# Convert to CallbackSetLRScheduler

Convert a `torch` scheduler generator to a `CallbackSetLRScheduler`.

## Usage

``` r
as_lr_scheduler(x, step_on_epoch)
```

## Arguments

- x:

  (`function`)  
  The `torch` scheduler generator defined using
  [`torch::lr_scheduler()`](https://torch.mlverse.org/docs/reference/lr_scheduler.html).

- step_on_epoch:

  (`logical(1)`)  
  Whether the scheduler steps after every epoch
