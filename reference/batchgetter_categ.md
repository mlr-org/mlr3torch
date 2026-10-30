# Batchgetter for Categorical data

Converts a data frame of categorical data into a long tensor by
converting the data to integers. No input checks are performed.

All columns are encoded with **1-based** codes, i.e. the values of a
feature with cardinality `k` are `1, ..., k`.

## Usage

``` r
batchgetter_categ(data, ...)
```

## Arguments

- data:

  (`data.table`)  
  `data.table` to be converted to a `tensor`.

- ...:

  (any)  
  Unused.
