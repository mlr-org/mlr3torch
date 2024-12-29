skip_on_cran()

test_that("CIFAR-10 works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("tsk_cifar10")
})