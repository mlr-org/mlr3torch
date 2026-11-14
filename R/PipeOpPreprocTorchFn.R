#' @title Custom Preprocessing
#' @name mlr_pipeops_trafo_fn
#'
#' @description
#' Applies a user-supplied function `fn` to (one or more) [`lazy_tensor`] columns of a task.
#' It is the preprocessing analogue of [`nn("fn")`][mlr_pipeops_nn_fn] and an alternative to creating a new class via
#' [`pipeop_preproc_torch()`].
#'
#' The function receives the tensor as its first argument and must return a tensor.
#' Its other arguments are the parameters of the `PipeOp`, see section Parameters.
#'
#' Being a `trafo_` operator, its `stages` parameter starts out as `"both"`, so the transformation is applied both
#' while training and when predicting.
#' For an augmentation that is only applied while training, use [`po("augment_fn")`][mlr_pipeops_augment_fn].
#' See [`PipeOpTaskPreprocTorch`] for the `stages` parameter and this naming convention.
#'
#' @section Parameters:
#' By default, the parameters are inferred as all arguments of `fn` except for the first one (and `...`).
#' They are untyped and tagged `"train"`, and those without a default value are tagged `"required"` as well.
#' To specify them more explicitly, e.g. to make them tunable, use the construction argument `param_set`.
#' In addition, there are the parameters `stages` and `affect_columns`, see [`PipeOpTaskPreprocTorch`].
#'
#' @section Shapes:
#' By default (`shapes_out = "infer"`), the output shape is inferred by applying `fn` to an empty tensor of the input
#' shape, see [`pipeop_preproc_torch()`].
#' This assumes that the output shape only depends on the input shape and the parameter values.
#' When this is not the case, e.g. because the output shape is random, set `shapes_out` to `NULL`, which declares
#' the output shape as unknown.
#' [`po("torch_ingress_ltnsr")`][mlr_pipeops_torch_ingress_ltnsr] then needs its `shape` parameter to be set.
#'
#' @section Input and Output Channels:
#' See [`PipeOpTaskPreprocTorch`].
#' @section State:
#' See [`PipeOpTaskPreprocTorch`].
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' task = tsk("lazy_iris")
#'
#' # The parameters are inferred from the arguments of the function
#' po_scale = po("trafo_fn", fn = function(x, a) x * a, a = 2)
#' po_scale$param_set
#'
#' task_scaled = po_scale$train(list(task))[[1L]]
#' materialize(task$data(1:2, "x"), rbind = TRUE)
#' materialize(task_scaled$data(1:2, "x"), rbind = TRUE)
#'
#' # The output shapes are inferred as well
#' po_square = po("trafo_fn", fn = function(x) torch_cat(list(x, x^2), dim = 2L))
#' po_square$shapes_out(list(c(NA, 4L)), stage = "train")
PipeOpPreprocTorchTrafoFn = R6Class("PipeOpPreprocTorchTrafoFn",
  inherit = PipeOpTaskPreprocTorch,
  public = list(
    #' @description Creates a new instance of this [`R6`][R6::R6Class] class.
    #' @param fn (`function`)\cr
    #'   The preprocessing function, which takes the tensor as its first argument and returns a tensor.
    #'   It must not modify its input in-place.
    #'   Unless `rowwise` is `TRUE`, the tensor includes the batch dimension.
    #' @param param_set ([`ParamSet`][paradox::ParamSet] or `NULL`)\cr
    #'   The parameter set, describing the arguments of `fn` except for the first one.
    #'   All parameters must be tagged `"train"` (and not `"predict"`).
    #'   If `NULL` (default), it is inferred from the arguments of `fn`, see section Parameters.
    #' @param shapes_out (`function` or `NULL` or `"infer"`)\cr
    #'   How to calculate the output shapes, see section Shapes.
    #'   A function is used as the private `.shapes_out()` method of [`PipeOpTaskPreprocTorch`],
    #'   i.e. it is called as `shapes_out(shapes_in, param_vals, task)` and must return a `list()` with one shape.
    #'   `NULL` declares the output shapes as unknown and `"infer"` (default) infers them.
    #' @param rowwise (`logical(1)`)\cr
    #'   Whether `fn` is applied to each element of a batch individually, i.e. without the batch dimension, instead
    #'   of to the whole batch. Initialized to `FALSE`.
    #' @template param_packages
    #' @template params_pipelines
    initialize = function(fn, id = "trafo_fn", param_vals = list(), param_set = NULL, shapes_out = "infer",
      rowwise = FALSE, packages = character(0)) {
      assert_function(fn)
      fn_args = formalArgs(fn)
      if (!length(fn_args)) {
        stopf("Argument 'fn' must take the tensor as its first argument, but takes no arguments.")
      }
      if (is.null(param_set)) {
        param_set = inferps(fn, ignore = fn_args[1L], required = TRUE)
      } else {
        assert_param_set(param_set)
        if ("..." %nin% fn_args) {
          assert_subset(param_set$ids(), fn_args[-1L], .var.name = "ids of param_set")
        }
        wrong_tags = names(Filter(function(tags) "train" %nin% tags || "predict" %in% tags, param_set$tags))
        if (length(wrong_tags)) {
          stopf("All parameters must be tagged 'train' and not 'predict', but this does not hold for: %s.",
            paste0(wrong_tags, collapse = ", "))
        }
      }
      if (!identical(shapes_out, "infer")) {
        assert_function(shapes_out, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE)
      }
      private$.shapes_out_fn = shapes_out

      super$initialize(
        fn = fn,
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        packages = packages,
        rowwise = rowwise,
        stages_init = private$.stages_init
      )
    }
  ),
  private = list(
    .stages_init = "both",
    .shapes_out_fn = NULL,
    .shapes_out = function(shapes_in, param_vals, task) {
      shapes_out = private$.shapes_out_fn
      if (is.null(shapes_out)) {
        return(list(NULL))
      }
      if (identical(shapes_out, "infer")) {
        return(infer_shapes(shapes_in = shapes_in, param_vals = param_vals, output_names = self$output$name,
          fn = self$fn, rowwise = self$rowwise, id = self$id))
      }
      shapes = shapes_out(shapes_in = shapes_in, param_vals = param_vals, task = task)
      assert_list(shapes, len = 1L, .var.name = "return value of 'shapes_out'")
      if (!is.null(shapes[[1L]])) {
        shapes = assert_shapes(shapes, coerce = TRUE)
      }
      shapes
    },
    .additional_phash_input = function() {
      # `rowwise` and `shapes_out` are as much a part of what this operator does as `fn` is
      c(super$.additional_phash_input(), list(self$rowwise, private$.shapes_out_fn))
    }
  )
)

#' @title Custom Augmentation
#' @name mlr_pipeops_augment_fn
#'
#' @description
#' Applies a user-supplied function `fn` to (one or more) [`lazy_tensor`] columns of a task.
#' This is the same as [`po("trafo_fn")`][mlr_pipeops_trafo_fn], see there for more information, except that,
#' being an `augment_` operator, its `stages` parameter starts out as `"train"`.
#' The augmentation is therefore applied while training and not when predicting.
#' Set `stages = "both"` to apply it in both phases.
#'
#' @inheritSection mlr_pipeops_trafo_fn Parameters
#' @inheritSection mlr_pipeops_trafo_fn Shapes
#' @inheritSection mlr_pipeops_trafo_fn Input and Output Channels
#' @inheritSection mlr_pipeops_trafo_fn State
#'
#' @export
#' @examplesIf torch::torch_is_installed()
#' task = tsk("lazy_iris")
#'
#' # Adds Gaussian noise to the features
#' po_noise = po("augment_fn", fn = function(x, sd) x + torch_randn_like(x) * sd, sd = 0.1)
#'
#' # The noise is added while training ...
#' task_train = po_noise$train(list(task))[[1L]]
#' materialize(task_train$data(1:2, "x"), rbind = TRUE)
#'
#' # ... but not when predicting
#' task_predict = po_noise$predict(list(task))[[1L]]
#' materialize(task_predict$data(1:2, "x"), rbind = TRUE)
PipeOpPreprocTorchAugmentFn = R6Class("PipeOpPreprocTorchAugmentFn",
  inherit = PipeOpPreprocTorchTrafoFn,
  public = list(
    #' @description Creates a new instance of this [`R6`][R6::R6Class] class.
    #' @param fn (`function`)\cr
    #'   The augmentation function, which takes the tensor as its first argument and returns a tensor.
    #'   It must not modify its input in-place.
    #'   Unless `rowwise` is `TRUE`, the tensor includes the batch dimension.
    #' @param param_set ([`ParamSet`][paradox::ParamSet] or `NULL`)\cr
    #'   The parameter set, describing the arguments of `fn` except for the first one.
    #'   All parameters must be tagged `"train"` (and not `"predict"`).
    #'   If `NULL` (default), it is inferred from the arguments of `fn`, see section Parameters.
    #' @param shapes_out (`function` or `NULL` or `"infer"`)\cr
    #'   How to calculate the output shapes, see section Shapes.
    #'   A function is used as the private `.shapes_out()` method of [`PipeOpTaskPreprocTorch`],
    #'   i.e. it is called as `shapes_out(shapes_in, param_vals, task)` and must return a `list()` with one shape.
    #'   `NULL` declares the output shapes as unknown and `"infer"` (default) infers them.
    #' @param rowwise (`logical(1)`)\cr
    #'   Whether `fn` is applied to each element of a batch individually, i.e. without the batch dimension, instead
    #'   of to the whole batch. Initialized to `FALSE`.
    #' @template param_packages
    #' @template params_pipelines
    initialize = function(fn, id = "augment_fn", param_vals = list(), param_set = NULL, shapes_out = "infer",
      rowwise = FALSE, packages = character(0)) {
      super$initialize(fn = fn, id = id, param_vals = param_vals, param_set = param_set, shapes_out = shapes_out,
        rowwise = rowwise, packages = packages)
    }
  ),
  private = list(
    .stages_init = "train"
  )
)

#' @include aaa.R utils.R shape.R PipeOpTaskPreprocTorch.R
register_po("trafo_fn", PipeOpPreprocTorchTrafoFn, metainf = list(fn = identity))
register_po("augment_fn", PipeOpPreprocTorchAugmentFn, metainf = list(fn = identity))
