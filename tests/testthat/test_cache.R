test_that("cache works if mlr3torch.cache is set to FALSE", {
  # if we disable caching, we expect the folder structure of the tempfile() to be:
  # raw:
  #   - data.csv (the "downloaded" data)
  withr::local_options(mlr3torch.cache = FALSE)

  dat = data.table(x = rnorm(1))

  test_constructor = function(path) {
    fwrite(dat, normalizePath(file.path(path, "data.csv"), mustWork = FALSE))
    return(dat)
  }

  dat1 = cached(test_constructor, "datasets", "test_data")

  expect_list(dat1, len = 2)
  expect_permutation(names(dat1), c("data", "path"))

  expect_equal(list.files(dat1$path), "raw")
  expect_equal(list.files(file.path(dat1$path, "raw")), "data.csv")
  expect_equal(dat, dat1$data)
})

test_that("cache works if mlr3torch.cache is set to a directory", {
  # If we enable caching, we expect the folder structure of cache_dir/datasets/test_data to be
  # raw:
  #   - data.csv (the "downloaded" data)
  # data.rds (the processed data)

  cache_dir = tempfile()
  withr::local_options(mlr3torch.cache = cache_dir)

  dat = data.table(x = rnorm(10), y = rnorm(10), row_id = 1:10)

  test_constructor = function(path) {
    fwrite(dat, file.path(path, "data.csv"))
    return(dat)
  }

  dat1 = cached(test_constructor, "datasets", "test_data")

  expect_list(dat1, len = 2)
  expect_permutation(names(dat1), c("data", "path"))

  expect_permutation(list.files(dat1$path), c("raw", "data.rds"))
  expect_equal(list.files(file.path(dat1$path, "raw")), "data.csv")
  expect_equal(dat, dat1$data)

  dat2 = cached(function(x) stop(), "datasets", "test_data")
  expect_equal(dat1$data, dat2$data)
  # /private/var and /var are symlinked and somehow different paths are returned on macOS
  expect_equal(normalizePath(dat1$path, mustWork = FALSE), normalizePath(dat2$path, mustWork = FALSE))
})

test_that("cache works if mlr3torch.cache is set to TRUE", {
  name = paste0(sample(letters, 20), collapse = "")
  withr::local_options(mlr3torch.cache = TRUE)
  cache_dir = get_cache_dir()
  withr::defer(unlink(file.path(cache_dir, "datasets", name)))

  dat = data.table(x = rnorm(1))

  test_constructor = function(path) {
    fwrite(dat, file.path(path, "data.csv"))
    return(dat)
  }

  dat1 = cached(test_constructor, "datasets", name)


  expect_permutation(list.files(dat1$path), c("raw", "data.rds"))
  expect_equal(list.files(file.path(dat1$path, "raw")), "data.csv")
  expect_equal(dat, dat1$data)

  dat2 = cached(function(x) stop(), "datasets", name)
  expect_equal(dat1, dat2)
})

test_that("no corrupt leftovers when construction throws", {
  cache_dir = tempfile()
  withr::local_options(mlr3torch.cache = cache_dir)
  test_constructor = function(path) {
    stop("ERROR")
  }

  expect_error(cached(test_constructor, "datasets", "test_data"), regexp = "ERROR")

  expect_true(!dir.exists(file.path(cache_dir, "datasets", "test_data")))
})

test_that("cache initialization and versioning are correct", {
  # here we mock what happens when a cache for a new subfolder (some_name)
  # is initialized with a version of 5.
  cache_dir = tempfile()
  withr::local_options(mlr3torch.cache = cache_dir)
  name = "some_name"
  CACHE$versions[[name]] = 5
  withr::defer({CACHE$versions[[name]] = NULL}) # nolint

  withr::local_options(mlr3torch.cache = cache_dir)
  dat = data.table(x = rnorm(1))

  test_constructor = function(path) {
    return(dat)
  }

  dat1 = cached(test_constructor, name, "test_data")
  cache_dir = normalizePath(cache_dir, mustWork = FALSE)

  # here the version should be 5
  cache_version = jsonlite::read_json(file.path(cache_dir, "version.json"))
  # the cache version is correctly written
  expect_true(cache_version[[name]] == 5)

  # the other cache version is left unchanged
  expect_true(cache_version$datasets == CACHE$versions$datasets)
  expect_true(normalizePath(cache_dir, mustWork = FALSE) %in% CACHE$initialized)
  # the subfolder is created
  assert_true(name %in% list.files(cache_dir))

  # ceate another file in a different cache subdirector (cache_dir/datasets/test_data)
  cached(test_constructor, "datasets", "test_data")

  # now we simulate what happens when we increment the cache version
  # first we need to remove the temporary cache_dir from initialized
  # as otherwise the version check is skipped
  CACHE$initialized = setdiff(CACHE$initialized, cache_dir)

  CACHE$versions[[name]] = 6

  dat2 = cached(test_constructor, name, "test_data1")
  cache_version = jsonlite::read_json(file.path(cache_dir, "version.json"))

  expect_true(cache_version[[name]] == 6)
  expect_permutation(list.files(cache_dir), c("datasets", "some_name", "version.json"))
})

test_that("correct error when directory was not initialized by mlr3torch", {
  dir = tempfile()
  dir.create(dir)
  withr::local_options(mlr3torch.cache = dir)

  expect_error(initialize_cache(dir), regexp = "not initialized")
})
