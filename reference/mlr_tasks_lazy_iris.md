# Iris Classification Task

A classification task for the popular
[datasets::iris](https://rdrr.io/r/datasets/iris.html) data set. Just
like the iris task, but the features are represented as one lazy tensor
column.

## Format

[R6::R6Class](https://r6.r-lib.org/reference/R6Class.html) inheriting
from
[mlr3::TaskClassif](https://mlr3.mlr-org.com/reference/TaskClassif.html).

## Source

<https://en.wikipedia.org/wiki/Iris_flower_data_set>

## Construction

    tsk("lazy_iris")

## Properties

- Task type: “classif”

- Properties: “multiclass”

- Has Missings: no

- Target: “Species”

- Features: “x”

- Data Dimension: 150x3

## References

Anderson E (1936). “The Species Problem in Iris.” *Annals of the
Missouri Botanical Garden*, **23**(3), 457.
[doi:10.2307/2394164](https://doi.org/10.2307/2394164) .

## Examples

``` r
task = tsk("lazy_iris")
task
#> 
#> ── <TaskClassif> (150x2): Iris Flowers ─────────────────────────────────────────
#> • Target: Species
#> • Target classes: setosa (33%), versicolor (33%), virginica (33%)
#> • Properties: multiclass
#> • Features (1):
#>   • lt (1): x
df = task$data()
materialize(df$x[1:6], rbind = TRUE)
#> torch_tensor
#>  5.1000  3.5000  1.4000  0.2000
#>  4.9000  3.0000  1.4000  0.2000
#>  4.7000  3.2000  1.3000  0.2000
#>  4.6000  3.1000  1.5000  0.2000
#>  5.0000  3.6000  1.4000  0.2000
#>  5.4000  3.9000  1.7000  0.4000
#> [ CPUFloatType{6,4} ]
```
