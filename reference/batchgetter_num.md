# Batchgetter for Numeric Data

Converts a data frame of numeric data into a float tensor by calling
[`as.matrix()`](https://rdrr.io/r/base/matrix.html). No input checks are
performed

## Usage

``` r
batchgetter_num(data, ...)
```

## Arguments

- data:

  ([`data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html))  
  `data.table` to be converted to a `tensor`.

- ...:

  (any)  
  Unused.
