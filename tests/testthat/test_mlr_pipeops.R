test_that("mlr_pipeops can be converted to a table", {
  expect_error(as.data.table(mlr_pipeops), regexp = NA)
  keys = names(po_register_env)
  mlr_pipeops2 = mlr_pipeops$clone()
  for (key in mlr_pipeops2$keys()) {
    if (key %nin% keys) {
      mlr_pipeops2$remove(key)
    }
  }
})
