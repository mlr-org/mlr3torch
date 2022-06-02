test_that("make_image_dataset works", {
  task = tsk("tiny_imagenet")
  expect_error(as_dataset(task), regexp = NA)
})
