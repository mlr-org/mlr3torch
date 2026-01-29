# Lazy Data Backend

This lazy data backend wraps a constructor that lazily creates another
backend, e.g. by downloading (and caching) some data from the internet.
This backend should be used, when some metadata of the backend is known
in advance and should be accessible before downloading the actual data.
When the backend is first constructed, it is verified that the provided
metadata was correct, otherwise an informative error message is thrown.
After the construction of the lazily constructed backend, calls like
`$data()`, `$missings()`, `$distinct()`, or `$hash()` are redirected to
it.

Information that is available before the backend is constructed is:

- `nrow` - The number of rows (set as the length of the `rownames`).

- `ncol` - The number of columns (provided via the `id` column of
  `col_info`).

- `colnames` - The column names.

- `rownames` - The row names.

- `col_info` - The column information, which can be obtained via
  [`mlr3::col_info()`](https://mlr3.mlr-org.com/reference/col_info.html).

Beware that accessing the backend's hash also contructs the backend.

Note that while in most cases the data contains
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
columns, this is not necessary and the naming of this class has nothing
to do with the
[`lazy_tensor`](https://mlr3torch.mlr-org.com/dev/reference/lazy_tensor.md)
data type.

**Important**

When the constructor generates
[`factor()`](https://rdrr.io/r/base/factor.html) variables it is
important that the ordering of the levels in data corresponds to the
ordering of the levels in the `col_info` argument.

## Super class

[`mlr3::DataBackend`](https://mlr3.mlr-org.com/reference/DataBackend.html)
-\> `DataBackendLazy`

## Active bindings

- `backend`:

  (`DataBackend`)  
  The wrapped backend that is lazily constructed when first accessed.

- `nrow`:

  (`integer(1)`)  
  Number of rows (observations).

- `ncol`:

  (`integer(1)`)  
  Number of columns (variables), including the primary key column.

- `rownames`:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Returns vector of all distinct row identifiers, i.e. the contents of
  the primary key column.

- `colnames`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Returns vector of all column names, including the primary key column.

- `is_constructed`:

  (`logical(1)`)  
  Whether the backend has already been constructed.

## Methods

### Public methods

- [`DataBackendLazy$new()`](#method-DataBackendLazy-new)

- [`DataBackendLazy$data()`](#method-DataBackendLazy-data)

- [`DataBackendLazy$head()`](#method-DataBackendLazy-head)

- [`DataBackendLazy$distinct()`](#method-DataBackendLazy-distinct)

- [`DataBackendLazy$missings()`](#method-DataBackendLazy-missings)

- [`DataBackendLazy$print()`](#method-DataBackendLazy-print)

Inherited methods

- [`mlr3::DataBackend$format()`](https://mlr3.mlr-org.com/reference/DataBackend.html#method-format)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    DataBackendLazy$new(constructor, rownames, col_info, primary_key)

#### Arguments

- `constructor`:

  (`function`)  
  A function with argument `backend` (the lazy backend), whose return
  value must be the actual backend. This function is called the first
  time the field `$backend` is accessed.

- `rownames`:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  The row names. Must be a permutation of the rownames of the lazily
  constructed backend.

- `col_info`:

  ([`data.table::data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html))  
  A data.table with columns `id`, `type` and `levels` containing the
  column id, type and levels. Note that the levels must be provided in
  the correct order.

- `primary_key`:

  (`character(1)`)  
  Name of the primary key column.

------------------------------------------------------------------------

### Method [`data()`](https://rdrr.io/r/utils/data.html)

Returns a slice of the data in the specified format. The rows must be
addressed as vector of primary key values, columns must be referred to
via column names. Queries for rows with no matching row id and queries
for columns with no matching column name are silently ignored. Rows are
guaranteed to be returned in the same order as `rows`, columns may be
returned in an arbitrary order. Duplicated row ids result in duplicated
rows, duplicated column names lead to an exception.

Accessing the data triggers the construction of the backend.

#### Usage

    DataBackendLazy$data(rows, cols)

#### Arguments

- `rows`:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Row indices.

- `cols`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Column names.

------------------------------------------------------------------------

### Method [`head()`](https://rdrr.io/r/utils/head.html)

Retrieve the first `n` rows. This triggers the construction of the
backend.

#### Usage

    DataBackendLazy$head(n = 6L)

#### Arguments

- `n`:

  (`integer(1)`)  
  Number of rows.

#### Returns

[`data.table::data.table()`](https://rdrr.io/pkg/data.table/man/data.table.html)
of the first `n` rows.

------------------------------------------------------------------------

### Method `distinct()`

Returns a named list of vectors of distinct values for each column
specified. If `na_rm` is `TRUE`, missing values are removed from the
returned vectors of distinct values. Non-existing rows and columns are
silently ignored.

This triggers the construction of the backend.

#### Usage

    DataBackendLazy$distinct(rows, cols, na_rm = TRUE)

#### Arguments

- `rows`:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Row indices.

- `cols`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Column names.

- `na_rm`:

  (`logical(1)`)  
  Whether to remove NAs or not.

#### Returns

Named [`list()`](https://rdrr.io/r/base/list.html) of distinct values.

------------------------------------------------------------------------

### Method `missings()`

Returns the number of missing values per column in the specified slice
of data. Non-existing rows and columns are silently ignored.

This triggers the construction of the backend.

#### Usage

    DataBackendLazy$missings(rows, cols)

#### Arguments

- `rows`:

  ([`integer()`](https://rdrr.io/r/base/integer.html))  
  Row indices.

- `cols`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  Column names.

#### Returns

Total of missing values per column (named
[`numeric()`](https://rdrr.io/r/base/numeric.html)).

------------------------------------------------------------------------

### Method [`print()`](https://rdrr.io/r/base/print.html)

Printer.

#### Usage

    DataBackendLazy$print()

## Examples

``` r
# We first define a backend constructor
constructor = function(backend) {
  cat("Data is constructed!\n")
  DataBackendDataTable$new(
    data.table(x = rnorm(10), y = rnorm(10), row_id = 1:10),
    primary_key = "row_id"
  )
}

# to wrap this backend constructor in a lazy backend, we need to provide the correct metadata for it
column_info = data.table(
  id = c("x", "y", "row_id"),
  type = c("numeric", "numeric", "integer"),
  levels = list(NULL, NULL, NULL)
)
backend_lazy = DataBackendLazy$new(
  constructor = constructor,
  rownames = 1:10,
  col_info = column_info,
  primary_key = "row_id"
)

# Note that the constructor is not called for the calls below
# as they can be read from the metadata
backend_lazy$nrow
#> [1] 10
backend_lazy$rownames
#>  [1]  1  2  3  4  5  6  7  8  9 10
backend_lazy$ncol
#> [1] 3
backend_lazy$colnames
#> [1] "x"      "y"      "row_id"
col_info(backend_lazy)
#>        id    type levels
#>    <char>  <char> <list>
#> 1:      x numeric [NULL]
#> 2:      y numeric [NULL]
#> 3: row_id integer [NULL]

# Only now the backend is constructed
backend_lazy$data(1, "x")
#> Data is constructed!
#>            x
#>        <num>
#> 1: 0.4248584
# Is the same as:
backend_lazy$backend$data(1, "x")
#>            x
#>        <num>
#> 1: 0.4248584
```
