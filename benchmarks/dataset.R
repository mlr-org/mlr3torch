devtools::load_all("~/mlr/mlr3")
devtools::load_all("~/mlr/mlr3torch")

lazy_iris = tsk("lazy_iris")
dt = lazy_iris$data(cols = "x")$x
dataset = dt[[1L]][[2L]]$dataset

dt = do.call(c, args = lapply(1:1000, function(i) dt))


profvis::profvis({materialize_internal(dt, rbind = TRUE)})
