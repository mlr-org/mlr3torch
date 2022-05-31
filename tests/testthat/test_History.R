test_that("History works", {
  run = function() {
    history = History$new(2, 1)
    history$append("accuracy", 0.8, "train")
    history$increment("train")
    history$append("accuracy", 0.9, "train")

    history$append("acc", 0.5, "valid")
    history$increment("valid")
    history$append("acc", 0.4, "valid")

    history$increment("epoch")
    history$append("accuracy", 0.95, "train")
    history$increment("train")
    history$append("accuracy", 0.95, "train")

    history$append("acc", 0.6, "valid")
    history$increment("valid")
    history$append("acc", 0.7, "valid")

    history
  }
  history = run()

  train_expected = list(
    accuracy = list(
      list(0.8, 0.9),
      list(0.95, 0.95)
    )
  )
  valid_expected = list(
    acc = list(
      list(0.5, 0.4),
      list(0.6, 0.7)
    )
  )
  expect_true(all.equal(history$train, train_expected))
  expect_true(all.equal(history$valid, valid_expected))
})
