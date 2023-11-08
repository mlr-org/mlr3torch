#' @title Base Class for Torch Preprocessing
#' @name mlr_pipeops_preproc_torch
#'
#' @description
#' This `PipeOp` can be used to preprocess (one or more) [`lazy_tensor`] columns contained in an [`mlr3::Task`].
#' The preprocessing function is specified as construction argument `fn` and additional arguments to this
#' function can be defined through the `PipeOp`'s parameter set.
#' The preprocessing is done per-column, i.e. the number lazy tensor output columns is equivalent
#' to the number of lazy tensor input columns.
#' The preprocessing that is applied during `$predict()` uses the parameters that were set during `$train()` and not
#' those that are set when performing the prediction.
#'
#' @section Inheriting:
#' In addition to specifying the construction arguments, you can overwrite the private `.shapes_out()` method:
#' * `.shapes_out(shapes_in, param_vals, task)`\cr
#'   (`list()`, `list(), `Task` or `NULL`) -> `list()`\cr
#'   This private method calculates the output shapes of the lazy tensor columns that are created from applying
#'   the preprocessing.
#'
#'   This private method only has the responsibility to calculate the output shapes for one input column, i.e. the
#'   input `shapes_in` can be assumed to have exactly one shape vector for which it must calculate the output shapes
#'   and return it as a `list()` of length 1.
#'   It can also be assumed that the shape is not `NULL` (unknown).
#'   It should also output a shape with exactly one NA in the first dimension or `NULL`if the output shape cannot be
#'   determined (e.g. because it is random).
#'
#' @template param_id
#' @template param_param_vals
#' @param fn (`function`)\cr
#'   The preprocessing function.
#' @param packages (`character()`)\cr
#'   The packages the preprocessing function depends on.
#' @param param_set ([`ParamSet`])\cr
#'   In case the function `fn` takes additional parameter besides a [`torch_tensor`] they can be
#'   specfied as parameters. None of the parameters can have the [`"predict"`] tag.
#'   Pay attention to set the correct `tags` for the parameters: if tag `"train"` is present,
#'   the preprocessing is applied during training and if tag `"predict"` is present, the preprocessing is applied
#'   during prediction (if `augment` is set to `FALSE`).
#' @param augment_init (`logical(1)` or `NULL`)\cr
#'   Initial value for the `augment` parameter.
#'   If it is `NULL`, there is no `augment` parameter.
#' @section Input and Output Channels:
#' See [`PipeOpTaskPreproc`].
#' @section State:
#' TODO:
#' @section Parameters:
#' In addition to the parameters inherited from [`PipeOpTaskPreproc`] as well as those specified during construction
#' as the argument `param_set` there are the following parameters:
#'
#' * `augment` :: `logical(1)`\cr
#'   (This parameter only exists of the `PipeOp`) is applied per column.
#'   Whether the to apply the preprocessing only during training (`TRUE`) or also during prediction (`FALSE`).
#'   This parameter is initialized to `FALSE`.
#'
#' @section Internals:
#' A [`PipeOpModule`] with one input and one output channel is created.
#' The pipeop simply applies the function `fn` to the input tensor while additionally
#' passing the paramter values (minus `augment` and `affect_columns`) to `fn`.
#' Then [`transform_lazy_tensor`] is called with the created [`PipeOpModule`] and the shapes obtained from the
#' `$shapes_out()` method of this `PipeOp`.
#'
#' @include PipeOpTorch.R
#' @export
#' @examples
#' # Creating a simple task
#' d = data.table(
#'   x1 = as_lazy_tensor(rnorm(10)),
#'   x2 = as_lazy_tensor(rnorm(10)),
#'   x3 = as_lazy_tensor(as.double(1:10)),
#'   y = rnorm(10)
#' )
#'
#' taskin = as_task_regr(d, target = "y")
#'
#' # Creating a simple preprocessing pipeop
#' po_simple = po("preproc_torch",
#'   # use mlr3misc::crate to get rid of unnecessary environment baggage
#'   fn = mlr3misc::crate(function(x, a) x + a),
#'   param_set = ps(a = p_int(tags = c("train", "predict", "required")))
#' )
#'
#' po_simple$param_set$set_values(
#'   a = 100,
#'   affect_columns = selector_name(c("x1", "x2")),
#'   augment = FALSE
#' )
#'
#' taskout_train = po_simple$train(list(taskin))[[1L]]
#' materialize(taskout_train$data(cols = c("x1", "x2")), rbind = TRUE)
#'
#' taskout_predict_noaug = po_simple$predict(list(taskin))[[1L]]
#' materialize(taskout_predict_noaug$data(cols = c("x1", "x2")), rbind = TRUE)
#'
#' po_simple$param_set$set_values(
#'   augment = TRUE
#' )
#'
#' # transformation is not applied
#' taskout_predict_aug = po_simple$predict(list(taskin))[[1L]]
#' materialize(taskout_predict_aug$data(cols = c("x1", "x2")), rbind = TRUE)
#'
#' # Creating a more complex preprocessing PipeOp
#'
#' PipeOpPreprocTorchPoly = R6::R6Class("PipeOpPreprocTorchPoly",
#'   inherit = PipeOpTaskPreprocTorch,
#'   public = list(
#'     initialize = function(id = "preproc_poly", param_vals = list()) {
#'       param_set = paradox::ps(
#'         n_degree = paradox::p_int(lower = 1L, tags = c("train", "required"))
#'       )
#'       param_set$set_values(
#'         n_degree = 1L
#'       )
#'       fn = mlr3misc::crate(function(x, n_degree) {
#'         torch::torch_cat(
#'           lapply(seq_len(n_degree), function(d) torch_pow(x, d)),
#'           dim = 2L
#'         )
#'       })
#'
#'       super$initialize(
#'         fn = fn,
#'         id = id,
#'         packages = character(0),
#'         param_vals = param_vals,
#'         param_set = param_set
#'       )
#'     }
#'   ),
#'   private = list(
#'     .shapes_out = function(shapes_in, param_vals, task) {
#'       # shapes_in is a list of length 1 containing the shapes
#'       checkmate::assert_true(length(shapes_in[[1L]]) == 2L)
#'       if (shapes_in[[1L]][2L] != 1L) {
#'         stop("Input shape must be (NA, 1)")
#'       }
#'       list(c(NA, param_vals$n_degree))
#'     }
#'   )
#' )
#'
#' po_poly = PipeOpPreprocTorchPoly$new(
#'   param_vals = list(n_degree = 3L, affect_columns = selector_name("x3"))
#' )
#'
#' po_poly$shapes_out(list(c(NA, 1L)))
#'
#' taskout = po_poly$train(list(taskin))[[1L]]
#' materialize(taskout$data(cols = "x3"), rbind = TRUE)
PipeOpTaskPreprocTorch = R6Class("PipeOpTaskPreprocTorch",
  inherit = PipeOpTaskPreproc,
  public = list(
    #' @description
    #' Creates a new instance of this [`R6`][R6::R6Class] class.
    initialize = function(fn, id = "preproc_torch", param_vals = list(), param_set = ps(), packages = character(0),
      augment_init = FALSE) {
      private$.fn = assert_function(fn)
      assert_flag(augment_init, null.ok = TRUE)

      param_set$add(ps(
        augment = p_lgl(tags = c("train", "required"))
      ))
      param_set$set_values(augment = augment_init)

      if (some(param_set$tags, function(tags) "predict" %in% tags)) {
        stopf("Parameter set cannot contain tags 'predict'.")
      }

      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        feature_types = "lazy_tensor",
        packages = packages
      )
    },
    #' @description
    #'  Calculates the output shapes that would result in applying the preprocessing to one or more
    #'  lazy tensor columns with the provided shape.
    #'  Names are ignored and only order matters.
    #'  It uses the parameeter values that are currently set.
    #' @param shapes_in (`list()` of (`integer()` or `NULL`))\cr
    #'   The input input shapes of the lazy tensors.
    #'   `NULL` indicates that the shape is unknown.
    #' @param stage (`character(1)`)\cr
    #'   The stage: either `"train"` or `"predict"`.
    #' @param task ([`Task`] or `NULL`)\cr
    #'   The task, which is very rarely needed.
    #' @return `list()` of (`integer()` or `NULL`)
    shapes_out = function(shapes_in, stage = NULL, task = NULL) {
      assert_choice(stage, c("train", "predict"))
      if (stage == "predict" && is.null(self$state$param_vals)) {
        stopf("Predict shapes can only be calculated after training the PipeOp.")
      }
      assert_r6(task, "Task", null.ok = TRUE)
      # either all shapes are NULL or none are NULL
      assert_shapes(shapes_in, named = FALSE, null_ok = TRUE)
      names(shapes_in) = NULL

      pv = if (stage == "train") {
        self$param_set$get_values(tags = "train")
      } else {
        self$state$param_vals
      }

      augment = pv$augment
      pv$augment = NULL
      pv$affect_columns = NULL

      if ((stage == "predict") && augment) {
        return(shapes_in)
      }

      shapes = map(shapes_in, function(s) {
        if (!is.null(s)) {
          private$.shapes_out(list(s), pv, task)[[1L]]
        }
      })

      return(shapes)
    }
  ),
  private = list(
    .train_task = function(task) {
      dt_columns = private$.select_cols(task)
      cols = dt_columns
      param_vals = self$param_set$get_values(tags = "train")

      if (!length(cols)) {
        self$state = list(dt_columns = dt_columns, param_vals = param_vals)
        return(task)
      }
      dt = task$data(cols = cols)

      self$state$param_vals = param_vals
      self$state$dt_columns = dt_columns

      dt = private$.transform(dt, task, param_vals, "train")

      task$select(setdiff(task$feature_names, cols))$cbind(dt)
    },
    .predict_task = function(task) {
      cols = self$state$dt_columns
      if (!length(cols)) {
        return(task)
      }
      dt = task$data(cols = cols)

      param_vals = self$state$param_vals
      dt = private$.transform(dt, task, param_vals, "predict")
      res = task$select(setdiff(task$feature_names, cols))$cbind(dt)

      res
    },
    .transform = function(dt, task, param_vals, stage) {
      # stage is "train" or "predict"
      param_vals$affect_columns = NULL
      trafo = private$.fn

      if (stage == "train") {
        param_vals$augment = NULL
        fn = if (length(param_vals)) {
          crate(function(x) {
            invoke(.f = trafo, x, .args = param_vals)
          }, param_vals, trafo, .parent = environment(trafo))
        } else {
          trafo
        }
      } else {
        augment = param_vals$augment
        param_vals$augment = NULL
        # augment can be NULL (in case the pipeop changes the output shape and hence augmentation is not supported)
        # or a logical(1)
        fn = if (isTRUE(augment)) {
          # We cannot simple add no module to the graph as we need to maintain the same graph-structure
          # as during training
          identity
        } else if (length(param_vals)) {
          crate(function(x) {
            invoke(.f = trafo, x, .args = param_vals)
          }, param_vals, trafo, .parent = environment(trafo))
        } else {
          trafo
        }
      }

      dt = map_dtc(dt, function(lt) {
        po_fn = PipeOpModule$new(
          # By randomizing the id we avoid ID clashes
          id = paste0(self$id, sample.int(.Machine$integer.max, 1)),
          module = fn,
          inname = self$input$name,
          outname = self$output$name,
          packages = self$packages
        )

        shape_before = lt$.pointer_shape
        shape_out = self$shapes_out(list(shape_before), stage = stage, task = task)[[1L]]

        shape_out_predict = if (stage == "train") {
          shape_in_predict = if (is.null(lt$.pointer_shape_predict)) shape_before
          # during `$train()` we also keep track of the shapes that would arise during train
          # This avoids that we first train a learner and then only notice during predict that the shapes
          # during the predict phase are wrong
          shape_out_predict = self$shapes_out(list(shape_before), stage = "predict", task = task)[[1L]]
        }
        x = transform_lazy_tensor(lt, po_fn, shape_out, shape_out_predict)

        # During "train" we will also calculate the shapes that would arise during prediction, making
        # some assumption that the task and the parameter values are the same.
        # This is then later used in PipeOpTorchModel **before** starting the training to verify that
        # a `$predict()` call is possible later.
        x
      })
      names(dt) = uniqueify(names(dt), setdiff(task$col_info$id, names(dt)))
      return(dt)
    },
    .additional_phash_input = function() {
      list(self$param_set$ids(), private$.fn, self$packages)
    },
    .shapes_out = function(shapes_in, param_vals, task) shapes_in,
    .fn = NULL
  )
)

#' @title Create Torch Preprocessing PipeOps
#' @description
#' Calls [`pipeop_preproc_torch_class`] and instantiates the instance with the given parameter values.
#' @inheritParams pipeop_preproc_torch_class
#' @param param_vals (`list()`)\cr
#'   The parameter values.
#' @export
pipeop_preproc_torch = function(id, fn, shapes_out, param_set = NULL, param_vals = list(),
  packages = character(0)) {
  pipeop_preproc_torch_class(
    id = id,
    fn = fn,
    shapes_out = shapes_out,
    param_set = param_set,
    packages = packages
    )$new(param_vals = param_vals)
}


create_ps_call = function(fn) {
  fmls = rlang::fn_fmls(fn)
  param_names = names(fmls)
  param_names = setdiff(param_names[-1L], "...")
  fmls = fmls[param_names]
  is_required = map(fmls, function(x) x == rlang::missing_arg())
  # Create an empty named list to store the arguments
  args = list()

  # Iterate through the elements of v and create expressions
  for (pname in param_names) {
    arg_name = as.name(pname)
    if (is_required[[pname]]) {
      arg_value = rlang::expr(p_uty(tags = c("train", "required")))
    } else {
      arg_value = rlang::expr(p_uty(tags = "train"))
    }
    args[[arg_name]] = arg_value
  }

  # Create the final language object
  result = rlang::expr(ps(!!!args))

  return(result)
}

#' @title Create Torch Preprocessing PipeOps
#' @description
#' Convenience functions to create objects of class [`PipeOpTaskPreprocTorch`] in a slightly more convenient way.
#' Start by reading the documentation of [`PipeOpTaskPreprocTorch`].
#' @template param_id
#' @param fn (`function`)\cr
#'   The preprocessing function.
#' @param shapes_out (`function` or `NULL` or "infer")\cr
#'   The private `.shapes_out(shapes_in, param_vals, task)` method of [`PipeOpTaskPreprocTorch`].
#'   If `NULL`, the pipeop does not change the output shapes.
#'   If `TRUE`, the output shape function is inferred and calculates the output shapes as follows:
#'   a) The input shapes are known:
#'      For an input shape of (NA, ...) a meta-tensor of shape (1, ...) is created and the preprocessing function is
#'      applied. Afterwards the batch dimension (1) is replaced with NA and the shape is returned.
#'   b) The input shapes are unknown (all NA):
#'      The dimension is returned as is.
#' @param param_set ([`ParamSet`] or `NULL`)\cr
#'   The parameter set.
#'   If this is left as `NULL` (default) the parameter set is inferred in the following way:
#'   All parameters but the first and `...` of `fn` are set as untyped parameters with tags 'train' and those that
#'   have nod default value are tagged as 'required' as well.
#'   Default values are not annotated.
#' @template param_param_vals
#' @template param_packages
#' @export
#' @returns An [`R6Class`][R6::R6Class] instance inheriting from [`PipeOpTaskPreprocTorch`]
#' @examples
#' po_example = pipeop_preproc_torch("preproc_example", function(x, a) x + a)
#' po_example
#' po_example$param_set
pipeop_preproc_torch_class = function(id, fn, shapes_out, param_set = NULL, packages = character(0)) {
  assert_function(fn)
  assert_true(length(formals(fn)) >= 1L)
  assert(
    check_function(shapes_out, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE),
    check_choice(shapes_out, "infer")
  )

  if (!is.null(param_set)) {
    assert_param_set(eval(param_set))
  } else {
    # This should add required tags where it is applicable
    param_set = create_ps_call(fn)
  }

  classname = paste0("PipeOpPreprocTorch", paste0(capitalize(strsplit(id, split = "_")[[1L]]), collapse = ""))
  # Note that we don't set default values

  if (identical(shapes_out, "infer")) {
    shapes_out = crate(function(shapes_in, param_vals, task) {
      sin = shapes_in[[1L]]
      if (all(is.na(sin))) {
        # This is only TRUE if the preprocessing pipeop does not change the number of dimensions.
        return(list(sin))
      }
      # set batch-dim to 1

      sin[1] = 1L
      tensor_in = invoke(torch_empty, .args = sin, device = torch_device("meta"))
      tensor_out = invoke(private$.fn, tensor_in, .args = param_vals)
      sout = dim(tensor_out)

      sout[1] = NA
      list(sout)
    })
  } else {
    assert_function(shapes_out, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE)
  }

  init_fun = crate(function(id = idname, param_vals = list()) {
    super$initialize(
      id = id,
      packages = packages,
      param_set = ps,
      param_vals = param_vals,
      fn = fn
    )
  })
  formals(init_fun)$id = id
  # param_set is already an expression
  body(init_fun)[[2]][[3]] = substitute(packages)
  body(init_fun)[[2]][[4]] = param_set
  body(init_fun)[[2]][[6]] = substitute(fn)

  Class = R6Class(classname,
    inherit = PipeOpTaskPreprocTorch,
    public = list(
      initialize = init_fun
    ),
    private = list(
      .shapes_out = shapes_out
    )
  )

  return(Class)
}

register_preproc = function(id, fn, param_set = NULL, shapes_out = NULL, packages = character(0)) {
  Class = pipeop_preproc_torch_class(id, fn, param_set = substitute(param_set), shapes_out = shapes_out, packages = packages)
  assign(Class$classname, Class, parent.frame())
  register_po(id, Class)
}


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


register_po("preproc_torch", PipeOpTaskPreprocTorch, metainf = list(fn = identity))
