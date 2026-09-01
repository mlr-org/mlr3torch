# Auto Device

Resolves the `device` parameter of a learner. `"auto"` becomes `"cuda"`
when a CUDA device is available and `"cpu"` otherwise; any other value
is returned unchanged, except that an explicit `"cuda"` without an
available CUDA device is an error rather than a failure deep inside
libtorch later on.

## Usage

``` r
auto_device(device = NULL)
```

## Arguments

- device:

  (`character(1)`)  
  The device. If not `NULL`, is returned as is.
