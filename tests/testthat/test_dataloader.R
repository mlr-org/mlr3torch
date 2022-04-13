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
  data_set = make_tabular_dataset(
    data = dat,
    target = "y",
    features = paste0("x_", c("lgl", "fct", "int", "num")),
    batch_size = 1,
    device = "cpu"
  )

  expect_r6(data_set, "dataset")
  expect_true(length(data_set) == 3L)
})
