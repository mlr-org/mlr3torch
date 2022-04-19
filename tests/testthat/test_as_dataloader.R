test_that("as_dataloader works", {
  task = tsk("iris")
  task
  expect_error(as_dataloader(task, device = "cpu", batch_size = 16L, shuffle = TRUE),
    regexp = NA
  )
  dl = as_dataloader(task, device = "cpu", batch_size = 16L)
  batch = dl$.iter()$.next()
  expect_equal(batch$y$shape, c(16, 1))
})
