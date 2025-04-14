test_that("DataBackendDataset works with basic numeric dataset", {
  skip_if_not_installed("torch")

  # Create a simple torch dataset with numeric features and targets
  ds = torch::dataset(
    initialize = function() {
      self$features = torch::torch_randn(50, 3)
      self$target = torch::torch_randn(50, 1)
    },
    .getitem = function(idx) {
      list(
        x = self$features[idx, ],
        y = self$target[idx, ]
      )
    },
    .length = function() {
      nrow(self$features)
    }
  )()

  # Create the backend
  backend = DataBackendDataset$new(
    dataset = ds,
    feature_names = c("x", "y"),
    feature_types = list(
      x = list(type = "numeric"),
      y = list(type = "numeric")
    )
  )

  # Test basic properties
  expect_r6(backend, classes = c("DataBackend", "DataBackendDataset"))
  expect_equal(backend$nrow, 50)
  expect_equal(backend$ncol, 3)  # x, y, and row_id
  expect_equal(backend$colnames, c("x", "y", "..row_id"))
  expect_equal(backend$rownames, 1:50)
  expect_equal(backend$primary_key, "..row_id")

  # Test data retrieval
  data = backend$data(rows = 1:5, cols = c("x", "y", "..row_id"))
  expect_data_table(data, nrows = 5, ncols = 3)
  expect_named(data, c("x", "y", "..row_id"))
  expect_equal(data[["..row_id"]], 1:5)
  expect_matrix(as.matrix(data[, "x"]), nrows = 5, ncols = 3)
  expect_vector(data[, "y"], len = 5)

  # Test head
  head_data = backend$head(3)
  expect_data_table(head_data, nrows = 3, ncols = 3)

  # Test distinct and missings
  distinct_vals = backend$distinct(rows = 1:10, cols = c("..row_id"))
  expect_list(distinct_vals, len = 1)
  expect_equal(distinct_vals[["..row_id"]], 1:10)

  missings = backend$missings(rows = 1:10, cols = c("x", "y"))
  expect_named(missings, c("x", "y"))
  expect_equal(missings[["x"]], 0)
  expect_equal(missings[["y"]], 0)

  # Test col_info
  ci = col_info(backend)
  expect_data_table(ci, nrows = 3)
  expect_equal(ci$id, c("x", "y", "..row_id"))
  expect_equal(ci$type, c("numeric", "numeric", "integer"))
})

test_that("DataBackendDataset works with getbatch method", {
  skip_if_not_installed("torch")

  # Create a dataset with getbatch instead of getitem
  ds = torch::dataset(
    initialize = function() {
      self$features = torch::torch_randn(30, 2)
      self$target = torch::torch_randn(30, 1)
    },
    .getbatch = function(idx) {
      list(
        x = self$features[idx, ],
        y = self$target[idx, ]
      )
    },
    .length = function() {
      nrow(self$features)
    }
  )()

  # Create the backend
  backend = DataBackendDataset$new(
    dataset = ds,
    feature_names = c("x", "y"),
    feature_types = list(
      x = list(type = "numeric"),
      y = list(type = "numeric")
    )
  )

  # Test data retrieval
  data = backend$data(rows = 1:5, cols = c("x", "y"))
  expect_data_table(data, nrows = 5, ncols = 2)
  expect_named(data, c("x", "y"))
  expect_matrix(as.matrix(data[, "x"]), nrows = 5, ncols = 2)
})

test_that("DataBackendDataset works with categorical features", {
  skip_if_not_installed("torch")

  # Create a simple torch dataset with categorical features
  ds = torch::dataset(
    initialize = function() {
      # Create 40 samples with 2 features:
      # - A categorical feature with 3 classes (0, 1, 2)
      # - A numeric feature
      self$cat_feature = torch::torch_randint(0, 3, size = c(40, 1))
      self$num_feature = torch::torch_randn(40, 2)
    },
    .getitem = function(idx) {
      list(
        cat = self$cat_feature[idx, ],
        num = self$num_feature[idx, ]
      )
    },
    .length = function() {
      nrow(self$cat_feature)
    }
  )()

  # Define levels for the categorical feature
  cat_levels = c("low", "medium", "high")

  # Create the backend
  backend = DataBackendDataset$new(
    dataset = ds,
    feature_names = c("cat", "num"),
    feature_types = list(
      cat = list(type = "factor", levels = cat_levels),
      num = list(type = "numeric")
    )
  )

  # Test data retrieval with categorical data
  data = backend$data(rows = 1:10, cols = c("cat", "num"))
  expect_data_table(data, nrows = 10, ncols = 2)
  expect_factor(data$cat, levels = cat_levels)
  expect_matrix(as.matrix(data[, "num"]), nrows = 10, ncols = 2)

  # Test col_info for categorical feature
  ci = col_info(backend)
  expect_equal(ci[id == "cat", "type"][[1]], "factor")
  expect_equal(ci[id == "cat", "levels"][[1]][[1]], cat_levels)
})

test_that("DataBackendDataset handles missing column requests correctly", {
  skip_if_not_installed("torch")

  # Create a simple torch dataset
  ds = torch::dataset(
    initialize = function() {
      self$x = torch::torch_randn(20, 3)
    },
    .getitem = function(idx) {
      list(x = self$x[idx, ])
    },
    .length = function() {
      nrow(self$x)
    }
  )()

  # Create the backend
  backend = DataBackendDataset$new(
    dataset = ds,
    feature_names = "x",
    feature_types = list(
      x = list(type = "numeric")
    )
  )

  # Test requesting non-existent columns
  data = backend$data(rows = 1:5, cols = c("x", "non_existent"))
  expect_data_table(data, nrows = 5, ncols = 1)
  expect_named(data, "x")

  # Test requesting only non-existent columns
  data = backend$data(rows = 1:5, cols = "non_existent")
  expect_data_table(data, nrows = 0, ncols = 0)

  # Test distinct with non-existent columns
  distinct_vals = backend$distinct(rows = 1:5, cols = c("x", "non_existent"))
  expect_list(distinct_vals, len = 2)
  expect_true(!is.null(distinct_vals$x))
  expect_null(distinct_vals$non_existent)

  # Test missings with non-existent columns
  missings = backend$missings(rows = 1:5, cols = c("x", "non_existent"))
  expect_named(missings, c("x", "non_existent"))
  expect_equal(missings[["non_existent"]], 0)
})

test_that("DataBackendDataset handles edge cases", {
  skip_if_not_installed("torch")

  # Create a very small dataset (1 row)
  ds = torch::dataset(
    initialize = function() {
      self$x = torch::torch_randn(1, 2)
    },
    .getitem = function(idx) {
      list(x = self$x[idx, ])
    },
    .length = function() {
      nrow(self$x)
    }
  )()

  # Create the backend
  backend = DataBackendDataset$new(
    dataset = ds,
    feature_names = "x",
    feature_types = list(
      x = list(type = "numeric")
    )
  )

  # Test with a single row dataset
  expect_equal(backend$nrow, 1)
  data = backend$data(rows = 1, cols = "x")
  expect_data_table(data, nrows = 1, ncols = 1)

  # Test with out-of-range row requests
  data = backend$data(rows = 100, cols = "x")
  expect_data_table(data, nrows = 0, ncols = 0)

  # Test with empty row requests
  data = backend$data(rows = integer(0), cols = "x")
  expect_data_table(data, nrows = 0, ncols = 0)
})

test_that("DataBackendDataset handles multi-dimensional tensors", {
  skip_if_not_installed("torch")

  # Create a dataset with a 3D tensor feature (e.g., image data)
  ds = torch::dataset(
    initialize = function() {
      # Create 10 samples with a 3D tensor (like a small image)
      self$images = torch::torch_randn(10, 3, 16, 16)  # 10 RGB images of 16x16
    },
    .getitem = function(idx) {
      list(image = self$images[idx, , , ])
    },
    .length = function() {
      self$images$size(1)
    }
  )()

  # Create the backend
  backend = DataBackendDataset$new(
    dataset = ds,
    feature_names = "image",
    feature_types = list(
      image = list(type = "numeric")
    )
  )

  # Test data retrieval with multi-dimensional data
  data = backend$data(rows = 1:3, cols = "image")
  expect_data_table(data, nrows = 3, ncols = 1)

  # Each row should contain a 3D array
  expect_true(is.list(data$image))
  expect_equal(length(data$image), 3)

  # Each element should be a 3D array (3x16x16)
  first_image = data$image[[1]]
  expect_equal(dim(first_image), c(3, 16, 16))
})
