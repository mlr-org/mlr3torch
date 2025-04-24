* Add `as_lazy_tensors()`
* Make it easier to se
* Fix the bug that the shapes are reported as unknown below and make the code easier.
  ```r
  ds = dataset("test",
    initialize = function() {
      self$x = torch_randn(100, 10)
      self$y = torch_randn(100, 1)
    },
    .getitem = function(i) {
      list(x = self$x[i, ], y = self$y[i])
    },
    .length = function() {
      nrow(self$x)
    }
  )()
  x_lt = as_lazy_tensor(ds, list(x = c(NA, 10), y = c(NA, 1)), input_map = "x")
  y_lt = as_lazy_tensor(ds, list(x = c(NA, 10), y = c(NA, 1)), input_map = "y")

  tbl = data.table(x = x_lt, y = y_lt)
  ```
* Add checks on usage of `DataBackendLazyTensors` in `task_dataset`
* Add optimization that truths values don't have to be loaded twice during resampling, i.e.
  once for making the predictions and once for retrieving the truth column.
* only allow caching converter columns in `DataBackendLazyTensors` (probably just remove the `cache` parameter)