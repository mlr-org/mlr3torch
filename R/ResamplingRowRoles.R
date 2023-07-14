#' @title Resampling Using Row Roles
#' @name mlr_resamplings_row_roles
#' @description
#' In mlr3 it is possible to manually set the row roles `use` and `test`.
#' This resampling sets the rows from `use` as the training ids and those from `test` as the test ids.
#' This can be useful for tasks like [tiny imagenet][mlr_tasks_tiny_imagenet] that come with predefined splits.
#'
#' @export
#' @examples
#' resampling = rsmp("row_roles")
#' resampling
#'
#' task = tsk("mtcars")
#' splits = partition(task)
#' task$row_roles$use = splits$train
#' task$row_roles$test = splits$test
#' rr = resample(task, lrn("regr.rpart"), resampling)
#' rr$score()
ResamplingRowRoles = R6Class("ResamplingRowRoles",
  inherit = Resampling,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize(
        id = "row_roles",
        param_set = ps(),
        duplicated_ids = TRUE,
        label = "Row Roles",
        man = "mlr3torch::mlr_resamplings_row_roles"
      )
    },
    #' @description
    #' Materializes fixed training and test splits for a given task and stores them in `$instance`.
    #'
    #' @param task ([Task])\cr
    #'   Task used for instantiation.
    #'
    #' @return
    #' Returns the object itself, but modified **by reference**.
    #' You need to explicitly `$clone()` the object beforehand if you want to keeps
    #' the object in its previous state.
    instantiate = function(task) {
      if (!is.null(task$strata) || !is.null(task$groups)) {
        stopf("With resampling 'row_roles', stratification and grouping cannot be ensured.")
      }
      if (!(length(task$row_roles$use) > 0 && length(task$row_roles$test) > 0)) {
        stopf("Row roles 'use' and 'test' must not be empty.")
      }

      self$instance = list(
        train = task$row_roles$use,
        test = task$row_roles$test
      )
      self$task_hash = task$hash
      self$task_nrow = task$nrow
      invisible(self)
    },
    #' @description
    #' Returns the row ids of the i-th train set.
    #'
    #' @param i (`integer(1)`)\cr
    #'   Iteration.
    #' @return (`integer()`) of row ids.
    train_set = function(i) {
      if (!self$is_instantiated) {
        stopf("Resampling '%s' has not been instantiated yet", self$id)
      }
      self$instance$train
    },
    #' @description
    #' Returns the row ids of the i-th test set.
    #'
    #' @param i (`integer(1)`)\cr
    #'   Iteration.
    #' @return (`integer()`) of row ids.
    test_set = function(i) {
      if (!self$is_instantiated) {
        stopf("Resampling '%s' has not been instantiated yet", self$id)
      }
      self$instance$test

    }
  ),
  active = list(
    #' @field iters (`integer(1)`)\cr
    #'   The number of iterations which are always one for this resampling.
    iters = function(rhs) {
      assert_ro_binding(rhs)
      1L
    }
  )
)

register_resampling("row_roles", function() ResamplingRowRoles$new())
