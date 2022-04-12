make_data = function() {
  data.table(
    x_lgl = c(FALSE, TRUE, FALSE),
    x_fct = as.factor(c("l1", "l2", "l1")),
    x_int = 3:1,
    x_num = c(0.5, 0.3, 1.2),
    ..row_id = 1:3,
    y = c(1, -1.2, 0.3)
  )
}

test_that("cat2tensor works", {
  dat = data.table(
    x_lgl = c(TRUE, FALSE, FALSE),
    x_fct = as.factor(c("l1", "l2", "l1"))
  )
  tensor_expected = torch_tensor(
    matrix(
      c(1L, 1L,
        0L, 2L,
        0L, 1L),
      byrow = TRUE,
      ncol = 2
    )
  )
  tensor = cat2tensor(dat, device = "cpu")
  expect_true(all(as.logical(tensor_expected == tensor)))
})

test_that("make_dataset works", {
  dat = make_data()
  fn = make_tabular_dataset(
    data = dat,
    target = "y",
    features = paste0("x_", c("lgl", "fct", "int", "num")),
    batch_size = 1,
    device = "cpu"
  )
  data_set = fn()
  expect_r6(data_set, "dataset")
  expect_true(length(data_set) == 3L)
  obs1 = data_set$.getitem(1)
  obs1_expected = list(
    x_num = torch_tensor(c(3, 0.5)),
    x_cat = torch_tensor(c(1L, 0L, 1L)),
    y = torch_tensor(1)
  )
  expect_true(torch_equal(obs1$x_cat, obs1_expected$x_cat))
  expect_true(torch_equal(obs1$x_num, obs1_expected$x_num))
  expect_true(torch_equal(obs1$y, obs1_expected$y))

  obs2 = data_set$.getitem(2)
  obs2_expected = list(
    x_num = torch_tensor(c(2, 0.3)),
    x_cat = torch_tensor(c(2L, 1L, 2L)),
    y = torch_tensor(-1.2)
  )
  expect_true(torch_equal(obs2$x_cat, obs2_expected$x_cat))
  expect_true(torch_equal(obs2$x_num, obs2_expected$x_num))
  expect_true(torch_equal(obs2$y, obs2_expected$y))

  obs3 = data_set$.getitem(3)
  obs3_expected = list(
    x_num = torch_tensor(c(1, 1.2)),
    x_cat = torch_tensor(c(1L, 0L, 1L)),
    y = torch_tensor(0.3)
  )
  expect_true(torch_equal(obs3$x_cat, obs3_expected$x_cat))
  expect_true(torch_equal(obs3$x_num, obs3_expected$x_num))
  expect_true(torch_equal(obs3$y, obs3_expected$y))
})


test_that("SequentialSampler works", {
  data = make_data()
  fn = make_dataset(
    3:2,
    data,
    target = "y",
    features = paste0("x_", c("chr", "lgl", "fct", "int", "num")),
    batch_size = 1,
    device = "cpu"
  )
  data_set = fn()
  sampler = SequentialSampler$new(data_set)
  data_loader = dataloader(
    data_set,
    sampler = sampler,
    batch_size = 1
  )
  batches = list()
  i = 1
  coro::loop(for (batch in data_loader) {
    batches[[i]] = batch
    i = i + 1
  })
  batch1_expected = list(
    x_num = torch_tensor(matrix(c(1, 1.2), nrow = 1)),
    x_cat = torch_tensor(matrix(c(1L, 0L, 1L), nrow = 1)),
    y = torch_tensor(matrix(0.3, nrow = 1))
  )
  expect_true(torch_equal(batches[[1]]$x_cat, batch1_expected$x_cat))
  expect_true(torch_equal(batches[[1]]$x_num, batch1_expected$x_num))
  expect_true(torch_equal(batches[[1]]$y, batch1_expected$y))
  batch2_expected = list(
    x_num = torch_tensor(matrix(c(2, 0.3), nrow = 1)),
    x_cat = torch_tensor(matrix(c(2L, 1L, 2L), nrow = 1)),
    y = torch_tensor(matrix(-1.2, nrow = 1))
  )
  expect_true(torch_equal(batches[[2]]$x_cat, batch2_expected$x_cat))
  expect_true(torch_equal(batches[[2]]$x_num, batch2_expected$x_num))
  expect_true(torch_equal(batches[[2]]$y, batch2_expected$y))
})

test_that("DataBackendTorchDataTable works", {
  data = make_data()
  backend = DataBackendTorchDataTable$new(data, "..row_id")
  data_loader = backend$dataloader(
    rows = 1:3,
    target = "y",
    features = c("x_chr", "x_int"),
    batch_size = 2,
    device = "cpu"
  )
  expect_r6(data_loader, "dataloader")
  batches = list()
  i = 1
  coro::loop(for (batch in data_loader) {
    batches[[i]] = batch
    i = i + 1
  })

  batch1_expected = list(
    x_cat = torch_tensor(matrix(c(1L, 2L), nrow = 2)),
    x_num = torch_tensor(matrix(c(3, 2), nrow = 2)),
    y = torch_tensor(matrix(c(1, -1.2), nrow = 2))
  )
  expect_true(torch_equal(batches[[1]]$x_cat, batch1_expected$x_cat))
  expect_true(torch_equal(batches[[1]]$x_num, batch1_expected$x_num))
  expect_true(torch_equal(batches[[1]]$y, batch1_expected$y))

  batch2_expected = list(
    x_cat = torch_tensor(matrix(c(1L), nrow = 1)),
    x_num = torch_tensor(matrix(c(1), nrow = 1)),
    y = torch_tensor(matrix(c(0.3), nrow = ))
  )
  expect_true(torch_equal(batches[[1]]$x_cat, batch1_expected$x_cat))
  expect_true(torch_equal(batches[[1]]$x_num, batch1_expected$x_num))
  expect_true(torch_equal(batches[[1]]$y, batch1_expected$y))
})

test_that("DataBackendTorchDataTable works with only numeric", {
  data = make_data()
  backend = DataBackendTorchDataTable$new(data, "..row_id")
  data_loader = backend$dataloader(rows = 2, target = "y", features = "x_num",
    batch_size = 1, device = "cpu"
  )
  batch1_expected = list(
    x_cat = NULL,
    x_num = torch_tensor(matrix(0.3), device = "cpu"),
    y = torch_tensor(matrix(-1.2), device = "cpu")
  )
  batches = list()
  i = 1
  coro::loop(for (batch in data_loader) {
    batches[[i]] = batch
    i = i + 1
  })
  expect_true(is.null(batches[["x_cat"]]))
  expect_true(torch_equal(batch1_expected[["x_num"]], batches[[1]]$x_num))
  expect_true(torch_equal(batch1_expected[["y"]], batches[[1]]$y))

})

test_that("DataBackendTorchDataTable works with only categorical", {
  data = make_data()
  backend = DataBackendTorchDataTable$new(data, "..row_id")
  data_loader = backend$dataloader(rows = 3, target = "y", features = "x_chr",
    batch_size = 1, device = "cpu"
  )
  batch1_expected = list(
    x_num = NULL,
    x_cat = torch_tensor(matrix(1L), device = "cpu"),
    y = torch_tensor(matrix(0.3), device = "cpu")
  )
  batches = list()
  i = 1
  coro::loop(for (batch in data_loader) {
    batches[[i]] = batch
    i = i + 1
  })
  expect_true(is.null(batches[["x_num"]]))
  expect_true(torch_equal(batch1_expected[["x_cat"]], batches[[1]]$x_cat))
  expect_true(torch_equal(batch1_expected[["y"]], batches[[1]]$y))

})

test_that("make_dataloader works with boston_housing", {
  dl = make_dataloader(tsk("boston_housing"), 1, "cpu")
  expect_error(dl$.iter()$.next(), regexp = NA)
})

test_that("make_dataloader works with mtcars", {
  dl = make_dataloader(tsk("mtcars"), 1, "cpu")
  expect_error(batch <<- dl$.iter()$.next(), regexp = NA)

})
