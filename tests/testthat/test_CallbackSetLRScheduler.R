test_that("autotest", {
  cb = t_clbk("lr_scheduler")
  expect_torch_callback(cb)
})

# for each lr scheduler
test_that("") {
  # set a random seed

  # train a network with just `torch` and the lr scheduler in question

  # the sequence of learning rates here is the expected values

  # check that the learning rates set by the callback (i.e. the optimizer's learning rate) match this
}