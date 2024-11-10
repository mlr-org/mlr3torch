test_that("autotest", {
  cb = t_clbk("freeze")
  expect_torch_callback(cb, check_man = TRUE)
})