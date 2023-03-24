test_that("mlr_pipeops can be converted to a table", {
  tbl = as.data.table(mlr_pipeops)
  expect_data_table(tbl)
})
