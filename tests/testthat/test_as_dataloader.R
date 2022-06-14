test_that("as_dataloader works", {
  task = tsk("iris")
  expect_error(as_dataloader(task, device = "cpu", batch_size = 16L, shuffle = TRUE),
    regexp = NA
  )
  dl = as_dataloader(task, device = "cpu", batch_size = 50L)
  batch = dl$.iter()$.next()
  expect_equal(batch$y$shape, 50)
})

test_that("shuffling works for dataloader", {
  task = tsk("iris")
  dl = as_dataloader(task, device = "cpu", batch_size = 50L, shuffle = TRUE)
  batch = dl$.iter()$.next()
  expect_equal(batch$y$shape, 50)
  expect_true(!all(as.array(batch$y) == 1))
})
