#' @Base Class for Torch Preprocessing
#' @name mlr_pipeops_task_preproc_torch
#'
#' @description
#' This `PipeOp` can be used to preprocess (one or more) [`lazy_tensor`] columns contained in an [`mlr3::Task`]
#' using the same preprocessing function.
#' The function that is applied is specified as construction argument `fn` and additional arguments to this
#' function can be defined through the `PipeOp`'s parameter set.
#'
#' @section Inheriting:
#' In addition to specifying the parameters `fn`, `packages` and `param_set` during construction you must also overwrite
#' the private `$.shapes_out()` method in case the preprocessing function changes the shape of the tensor.
#' See the description of [`PipeOpTorch`] how to implement this method.
#'
#' @param fn (`function`)\cr
#'   The preprocessing function.
#' @param packages (`character()`)\cr
#'   The packages the preprocessing function depends on.
#' @param param_set ([`ParamSet`])\cr
#'   In case the function `fn` takes additional parameter besides a [`torch_tensor()`] they can be
#'   specfied as parameters. Pay attention to set the correct `tags` for the parameters: if tag `"train"` is present,
#'   the preprocessing is applied during training and if tag `"predict"` is present, the preprocessing is applied
#'   during prediction (if `augment` is set to `FALSE`).
#'
#' @section Input and Output Channels:
#' See [`PipeOpTaskPreproc`].
#' Note that the `PipeOp` currently expects exactly one [`lazy_tensor`] column.
#' @section State
#' See [`PipeOpTaskPreproc`].
#' @section Parameters:
#' In addition to the parameters inherited from [`PipeOpTaskPreproc`] as well as those specified during construction
#' as the argument `param_set` there is:
#'
#' * `augment` :: `logical(1)`\cr
#'   Whether the to apply the preprocessing only during training (`TRUE`) or also during prediction (`FALSE`).
#'   This parameter is initialized to `FALSE`.
#'
#' @section Internals:
#'
#' A [`PipeOpModule`] with one input and one output channel is created.
#' The pipeop simply applies the function `fn` to the input tensor while additionally
#' passing the paramter values (minus `augment` and `affect_columns`) to `fn`.
#' Then [`transform_lazy_tensor`] is called with the created [`PipeOpModule`] and the shapes obtained from the
#' `$shapes_out()` method of this `PipeOp`.
#'
#' @include PipeOpTorch.R
#' @export
#' @examples
if (FALSE) {
  PipeOpPreprocTorchPoly = R6::R6Class("PipeOpPreprocTorchPoly",
    inherit = PipeOpTaskPreprocTorch,
    public = list(
      initialize = function(id = "preproc_poly", param_vals = list()) {
        param_set = paradox::ps(
          n_degree = paradox::p_int(lower = 1L, tags = c("train", "predict", "required"))
        )
        param_set$set_values(
          n_degree = 1L
        )
        fn = mlr3misc::crate(function(x, n_degree) {
          torch::torch_cat(
            lapply(seq_len(n_degree), function(d) torch_pow(x, d)),
            dim = 2L
          )
        })

        super$initialize(
          fn = fn,
          id = id,
          packages = character(0),
          param_vals = param_vals,
          param_set = param_set
        )
      }
    ),
    private = list(
      .shapes_out = function(shapes_in, param_vals, task) {
        # shapes_in is a list of length 1 containing the shapes
        checkmate::assert_true(length(shapes_in[[1L]]) == 2L)
        if (shapes_in[[1L]][2L] != 1L) {
          stop("Input shape must be (NA, 1)")
        }
        list(c(NA, param_vals$n_degree))
      }
    )
  )

  po_poly = PipeOpPreprocTorchPoly$new(
    param_vals = list(n_degree = 2L)
  )

  po_poly$shapes_out(c(NA, 1L))

  d = data.table(
    x1 = as_lazy_tensor(rnorm(10)),
    x2 = as_lazy_tensor(rnorm(10)),
    y = rnorm(10)
  )

  task = as_task_regr(d, target = "y")

  po_pol

  po_preproc = po("preproc_torch",
    # use mlr3misc::crate to get rid of unnecessary environment baggage
    fn = mlr3misc::crate(function(x, a) x + a),
    param_set = ps(a = p_int(tags = c("train", "predict", "required")))
  )

  po_preproc$param_set$set_values(
    a = 100
  )

  taskout_train = po_preproc$train(list(task))[[1L]]

  materialize(taskout_train$data(cols = c("x1", "x2")))

}
PipeOpTaskPreprocTorch = R6Class("PipeOpTaskPreprocTorch",
  inherit = PipeOpTaskPreproc,
  public = list(
    initialize = function(fn, id = "preproc_torch", param_vals = list(), param_set = ps(), packages = character(0)) {
      private$.fn = assert_function(fn, null.ok = FALSE)

      if ("augment" %in% param_set$ids()) {
        stopf("Parameter name 'augment' is reserved and cannot be used.")
      }
      if (is.null(private$.shapes_out)) {
        param_set$add(ps(
          augment = p_lgl(tags = c("predict", "required"))
        ))
        param_set$set_values(augment = FALSE)
      }

      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        feature_types = "lazy_tensor"
      )
    },
    #' @description
    #'  Calculates the output shape after applying the preprocessing step to a lazy tensor vector.
    #' @param shapes_in (`list()` of `integer()` or `integer()`)\cr
    #'   The input input shape of the lazy tensor, which must either be a `list()` of length 1 containing the
    #'   shape of the lazy tensor (first dimension must be `NA` for the batch) or the shape itself.
    #' @param task ([`Task`] or `NULL`)\cr
    #'   The task, which is very rarely used.
    #' @return
    #'  A named `list()` containing the output shape. The name is the name of the output channel.
    shapes_out = function(shapes_in, task = NULL) {
      assert_r6(task, "Task", null.ok = TRUE)
      if (is.numeric(shapes_in)) shapes_in = list(shapes_in)
      if (identical(self$input$name, "...")) {
        assert_list(shapes_in, min.len = 1, types = "numeric")
      } else {
        assert_list(shapes_in, len = nrow(self$input), types = "numeric")
      }

      s = if (is.null(private$.shapes_out)) {
        shapes_in
      } else {
        pv = self$param_set$get_values()
        private$.shapes_out(shapes_in, pv, task = task)
      }

      set_names(s, self$output$name)
    }
  ),
  private = list(
    .train_task = function(task) {
      dt_columns = private$.select_cols(task)
      cols = dt_columns
      if (!length(cols)) {
        self$state = list(dt_columns = dt_columns)
        return(task)
      }
      dt = task$data(cols = cols)

      param_vals = self$param_set$get_values(tags = "train")
      param_vals$affect_columns = NULL

      trafo = private$.fn

      fn = if (length(param_vals)) {
        crate(function(x) {
          invoke(.f = trafo, x, .args = param_vals)
        }, param_vals, trafo, .parent = environment(trafo))
      } else {
        trafo
      }

      dt = private$.transform(dt, fn, task)

      self$state$dt_columns = dt_columns
      task$select(setdiff(task$feature_names, cols))$cbind(dt)
    },
    .predict_task = function(task) {
      cols = self$state$dt_columns
      if (!length(cols)) {
        return(task)
      }
      dt = task$data(cols = cols)
      param_vals = self$param_set$get_values(tags = "predict")
      augment = param_vals$augment
      param_vals$augment = NULL

      # augment can be NULL (in case the pipeopf changes the output shape and hence augmentation is not supported)
      # or a logical(1)
      fn = if (isTRUE(augment)) {
        # We cannot simple add no module to the graph as we need to maintain the same graph-structure
        # as during training
        identity
      } else {
        trafo = private$.fn
        crate(function(x) {
          invoke(.f = trafo, x, .args = param_vals)
        }, param_vals, trafo, .parent = environment(trafo))
      }

      dt = private$.transform(dt, fn, task)
      task$select(setdiff(task$feature_names, cols))$cbind(dt)
    },
    .transform = function(dt, fn, task) {
      dt = map_dtc(dt, function(lt) {
        # the ID of the `PipeOp` does not matter.
        # By randomizing it we avoid ID clashes

        po_fn = PipeOpModule$new(
          id = paste0(self$id, sample.int(.Machine$integer.max, 1)),
          module = fn,
          inname = self$input$name,
          outname = self$output$name,
          packages = self$packages
        )
        shapes_out = self$shapes_out(lt$.pointer_shape, task)
        transform_lazy_tensor(lt, po_fn, shapes_out[[1L]])
      })
      return(dt)
    },
    .fn = NULL,
    .additional_phash_input = function() {
      list(self$param_set$ids(), private$.fn, self$packages)
    }
  )
)

pipeop_preproc_torch = function(fn, id, param_set = NULL) {
  # TODO


# Where I was:
# PipeOpTaskPreprocTorch and PipeOpTorchTrafo:
# their shapes_out method should be slightly different, i.e.
# --> How to implement this?
# 1. In register_lazy I want to specify the shapes_out method only once.
# 2. PipeOpTorch expects list()s but ...TaskPreproc should probably input and output shape
# vectors
#
# Other thigns:
  # I want to provide helper functions to create these pipeops for:
  # * preproc
  # * trafo
  # * pipeop torch
# Other things:
  # * autotest for trafo and preproc
  # * implement some common trafos and preprocs
  #
  #


register_po("preproc_torch", PipeOpTaskPreprocTorch)

