test_that("DataBackendDataset works", {
  test_dataset = torch::dataset(
    initialize = function(n = 20, shapes = list(feature1 = 3, target = 1)) {
      self$n = n
      self$tensors = lapply(shapes, function(shape) {
        torch_randn(n, shape)
      })
    },
    .getitem = function(index) {
      lapply(self$tensors, function(feature) feature[index,])
    },
    .length = function() {
      self$n
    }
  )

  ds = test_dataset()
  indices = 1:2

  ds_sub = datasubset(
    test_dataset(),
    rows = indices,
    cols = "target",
    device = "cpu"
  )
  ds_sub$.getitem(1)
  expect_equal(length(indices), length(ds_sub))
  length(ds_sub)

  expect_equal(names(ds_sub$.getitem(1)), "target")

  backend = DataBackendDataset$new(
    data = ds,
    colnames = c("feature1", "target"),
    primary_key = "..my_row_id"
  )
  backend

  task = TaskClassifTorch$new(backend, "test_task", "target")
  dataset = task$data()
  expect_class(dataset, "datasubset")

  learner = lrn("classif.mlp", batch_size = 10, device = "cpu", epochs = 1)

  learner$train(task)



})
