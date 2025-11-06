# Concatenates multiple tensors

Concatenates multiple tensors on a given dimension. No broadcasting
rules are applied here, you must reshape the tensors before to have the
same shape.

## Usage

``` r
nn_merge_cat(dim = -1)
```

## Arguments

- dim:

  (`integer(1)`)  
  The dimension for the concatenation.
