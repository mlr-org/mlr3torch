test_that("DataBackendDataset", {
  ds = dataset(
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

  tbl = as_lazy_tensors(ds, list(x = c(NA, 10), y = c(NA, 1)))
  tbl$row_id = 1:100

  be = DataBackendLazyTensors$new(tbl, primary_key = "row_id", converter = list(y = as.numeric), cache = "y")

  expect_data_backend(be)

  be$data(1, "y")

  be$data(2, c("x", "y"))

  be$head()


  withr::with_options(list(mlr3torch.data_loading = TRUE), {
    be$data(1, c("x", "y"))
  })

  learner = lrn("regr.mlp", batch_size = 32, epochs = 1)

  task = as_task_regr(be, target = "y")
  learner$train(task)
})

test_that("mlp works with it", {
  learner = lrn("classif.mlp")
})
