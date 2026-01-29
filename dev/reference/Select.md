# Selector Functions for Character Vectors

A `Select` function subsets a character vector. They are used by the
callback `CallbackSetUnfreeze` to select parameters to freeze or
unfreeze during training.

## Usage

``` r
select_all()

select_none()

select_grep(pattern, ignore.case = FALSE, perl = FALSE, fixed = FALSE)

select_name(param_names, assert_present = TRUE)

select_invert(select)
```

## Arguments

- pattern:

  See [`grep()`](https://rdrr.io/r/base/grep.html)

- ignore.case:

  See [`grep()`](https://rdrr.io/r/base/grep.html)

- perl:

  See [`grep()`](https://rdrr.io/r/base/grep.html)

- fixed:

  See [`grep()`](https://rdrr.io/r/base/grep.html)

- param_names:

  The names of the parameters that you want to select

- assert_present:

  Whether to check that `param_names` is a subset of the full vector of
  names

- select:

  A `Select`

## Functions

- `select_all()`: `select_all` selects all elements

- `select_none()`: `select_none` selects no elements

- `select_grep()`: `select_grep` selects elements with names matching a
  regular expression

- `select_name()`: `select_name` selects elements with names matching
  the given names

- `select_invert()`: `select_invert` selects the elements NOT selected
  by the given selector

## Examples

``` r
select_all()(c("a", "b"))
#> [1] "a" "b"
select_none()(c("a", "b"))
#> character(0)
select_grep("b$")(c("ab", "ac"))
#> [1] "ab"
select_name("a")(c("a", "b"))
#> [1] "a"
select_invert(select_all())(c("a", "b"))
#> character(0)
```
