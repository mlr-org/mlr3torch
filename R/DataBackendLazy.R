##' @title Lazy Data Backend
##' @description
##' This lazy data backend wraps a constructor that lazily creates another backend, usually by downloading
##' (and caching) some data from the internet.
##' This backend should be used, when some metadata of the backend is known in advance and access should be possible
##' before downloading the actual data.
##' This includes the `nrow`, `ncol`, `rownames` and `colnames` which must be provided in the constructor.
##' It is up to the user to ensure that these values are correct.
##' Checks on their correctness are nonetheless performed by default after calling the constructor, but can be disabled
##' in case one is certain that the information is correct.
##'
##' @param constructor (`function()`)\cr
##'   A function, wither with no arguments, or with with arguments `path` and `cache_version`.
##'   This function must return a backend and is called once accessing `$data()` the first time.
##'   If the arguments `path` and `cache_version` exist, the addtional arguments `path` and `cache_version` must
##'   also be specified in the constructor which are passed to the function.
##'   Argument `path` is a path relative to the mlr3torch's cache folder, while `cache_version` is used to
##'   ensure that the cache is valid (incrementing the cache version will lead to an invalid cache).
##' @param nrow (`integer(1)`)\cr
##'   The number of rows of the lazily constructed backend.
##' @param ncol (`integer(1)`)\cr
##'   The number of columns of the lazily constructed backend.
##' @param rownames (`integer()`)\cr
##'   Unique row identifiers of the .
##' @param colnames (`character()`)\cr
##'   The unique column names.
##' @param path (`character(1)`)\cr
##'   A path relative to mlr3torch's cache folder where to cache the downloaded data.
##'   If `NULL` (default) no caching is done, but the `constructor` must not require the `path` argument.
##' @param cache_version (`integer(1)`)\cr
##'   The cache vevrsion. If `NULL` (default), no caching is done, but the `constructor` must not require the `cache_version`
##'   argument.
##'
#DataBackendLazy = R6Class("DataBackendLazy",
#  inherit = DataBackend,
#  cloneable = FALSE,
#  public = list(
#    initialize = function(
#      constructor,
#      nrow,
#      ncol,
#      rownames,
#      colnames,
#      path = NULL,
#      cache_version = NULL,
#      check_backend = TRUE
#      ) {
#      private$.ncol = assert_int(ncol, lower = 2)
#      private$.nrow = assert_int(nrow, lower = 1)
#      private$.rownames = assert_integerish(rownames, unique = TRUE, any.missing = FALSE, len = nrow)
#      private$.colnames = assert_names(colnames, type = "unique")
#      assert(
#        check_function(constructor, args = c("path", "cache_version")),
#        check_function(constructor, nargs = 0)
#      )
#      private$.constructor = constructor
#      private$.path = assert_string(path)
#      private$.cache_version = assert_int(cache_version)
#      private$.check_backend = assert_flag(check_backend)
#    },
#    data = function(rows, cols, data_format = "data.table") {
#      if (is.null(private$.backend)) {
#        # TODO: Do we really want these checks here?
#        # alternatively we can also simply NOT export DataBackendLazy
#        private$.backend = assert_backend(private$.constructor())
#        if ()
#        assert_subset(self$data_types, private$.backend$data_types)
#        assert_permutation(private$.backend$)
#      }
#      private$.backend(rows, cols, data_format)
#    }
#  ),
#  active = list(
#    nrow = function(rhs) {
#      assert_ro_binding(rhs)
#      private$.nrow
#    },
#    ncol = function(rhs) {
#      assert_ro_binding(rhs)
#      private$.ncol
#    },
#    rownames = function(rhs) {
#      assert_ro_binding(rhs)
#      private$.rownames
#    },
#    colnames = function(rhs) {
#      assert_ro_binding(rhs)
#      private$.colnames
#    }
#  ),
#  private = list(
#    .calculate_hash = function() {
#      calculate_hash(self$data)
#    },
#    .constructor = NULL,
#    .backend = NULL,
#    .cache_folder = NULL,
#    .cache_version = NULL,
#    .check_backend = NULL
#  )
#)
#
#if (FALSE) {
#
#  mnist_constructor = function() {
#    dir = get_cache_dir()
#
#    dataset_train = torchvision::mnist_dataset(dir)
#    n_train = nrow(dataset_train$data)
#    dataset_test = torchvision::mnist_dataset(dir)
#    n_test = nrow(dataset_test$data)
#
#    images_train = lapply(seq_len(n_train), function(i) array(dataset_train$data[i, ,], dim = c(1, 28, 28)))
#    images_test = lapply(seq_len(n_test), function(i) array(dataset_test$data[i, ,], dim = c(1, 28, 28)))
#
#    images = image_vector(c(images_train, images_test))
#
#    targets = c(dataset_train$targets, dataset_test$test_target)
#    targets = factor(targets, levels = 1:10, labels = c("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
#
#    dt = data.table(
#      image = images,
#      letter = targets
#    )
#
#    as_data_backend(dt)
#  }
#
#  backend_mnist = DataBackendLazy$new(
#    constructor = mnist_constructor,
#    nrow =
#
#
#  )
#
#
#  task = as_task_classif(dt, target = "letter")
#
#
#  imges = dataset$data
#  targets = dataset$
#
#  dat
#}
