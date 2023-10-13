#' @export
PipeOpTaskPreprocLazy = R6Class("PipeOpTaskPreprocLazy",
  inherit = PipeOpTaskPreproc,
  public = list(
    initialize = function(fn, id = "lazy_preproc", param_vals = list(), param_set = ps(), packages = character(0)) {
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
        feature_types = "lazy_tensor"
      )
    }
  ),
  private = list(
    .train_dt = function(dt, levels, target) {
      if (ncol(dt) != 1L) {
        # Check only during train as this will ensure it also holds during predict
        stop("Can only use PipeOpTorchLazyTrafo on tasks with exactly one lazy tensor column.")
      }
      param_vals = self$param_set$get_values(tags = "train")

      trafo = private$.fn

      fn = if (length(param_vals)) {
        crate(function(x) {
          invoke(.f = trafo, x, .args = param_vals)
        }, param_vals, trafo, .parent = topenv())
      } else {
        trafo
      }

      self$state = list()

      private$.transform_dt(dt, fn)
    },
    .predict_dt = function(dt, levels) {
      param_vals = self$param_set$get_values(tags = "predict")
      augment = param_vals$augment
      param_vals$augment = NULL

      # augment can be NULL (in case the pipeopf changes the output shape and hence augmentation is not supported)
      # or a logical(1)
      fn = if (isTRUE(augment)) {
        identity
        } else {
          trafo = private$.fn
        crate(function(x) {
          invoke(.f = trafo, x, .args = param_vals)
        }, param_vals, trafo, .parent = topenv())
      }

      private$.transform_dt(dt, fn)
    },
    .transform_dt = function(dt, fn) {
      po_fn = PipeOpModule$new(
        id = self$id,
        module = fn,
        inname = self$input$name,
        outname = self$output$name,
        packages = self$packages
      )

      lt = dt[[1L]]

      shapes_in = attr(lt[[1L]], "data_descriptor")$.pointer_shape

      shapes_out = self$shapes_out(shapes_in, intask)
      dt[[1L]] = transform_lazy_tensor(lt, po_fn, shapes_out[[1L]])
      return(dt)
    },
    .fn = NULL,
    .additional_phash_input = function() {
      list(self$param_set$ids(), private$.fn, self$packages)
    }
  )
)
