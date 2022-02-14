test_that("Reflections are complete", {
  known_loss <- c(torch_reflections$loss$classif, torch_reflections$loss$regr)

  # Get exported losses from torch, format "nn_<name>_loss"
  torch_loss <- getNamespaceExports("torch") |>
    stringi::stri_subset(regex = "^nn\\_.*loss$") |>
    stringi::stri_replace("", regex = "^nn\\_") |>
    stringi::stri_replace("", regex = "\\_loss$")

  expect_identical(sort(known_loss), sort(torch_loss))

  known_optimizers <- torch_reflections$optimizer

  # Get exported optimizers from torch, format "optim_<name>"
  torch_optimizers <- getNamespaceExports("torch") |>
    stringi::stri_subset(regex = "^optim\\_") |>
    stringi::stri_replace("", regex = "^optim\\_")

  # Manually add madgrad::optim_madgrad
  torch_optimizers <- c(torch_optimizers, "madgrad")

  expect_identical(sort(known_optimizers), sort(torch_optimizers))
})
