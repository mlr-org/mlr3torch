test_that("as_dataloader works", {
  task = tsk("iris")
  task
  expect_error(as_dataloader(task, device = "cpu", batch_size = 16L, shuffle = TRUE),
    regexp = NA
  )
  dl = as_dataloader(task, device = "cpu", batch_size = 16L)

  loop(for (batch in dl) {
    print(batch)
  })
})
