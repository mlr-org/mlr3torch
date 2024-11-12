skip_on_cran()

test_that("melanoma task works", {
  withr::local_options(mlr3torch.cache = TRUE)
  task = tsk("melanoma")
  # this makes the test faster
  # task$filter(1:10)
  expect_equal(task$id, "melanoma")
  expect_equal(task$label, "Melanoma classification")
  expect_equal(task$feature_names, c("sex", "anatom_site_general_challenge", "age_approx", "image"))
  expect_equal(task$target_names, "benign_malignant")
  expect_equal(task$man, "mlr3torch::mlr_tasks_melanoma")
  expect_equal(task$properties, c("twoclass", "groups"))

  task$data()

  expect_true("datasets--carsonzhang--ISIC_2020_small" %in% list.files(file.path(get_cache_dir(), "datasets", "melanoma", "raw")))
  expect_true("data.rds" %in% list.files(file.path(get_cache_dir(), "datasets", "melanoma")))
  expect_equal(task$backend$nrow, 32701 + 10982)
  expect_equal(task$backend$ncol, 5)
})