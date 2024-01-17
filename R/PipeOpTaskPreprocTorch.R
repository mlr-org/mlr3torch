#' @title Base Class for Lazy Tensor Preprocessing
#' @name mlr_pipeops_preproc_torch
#'
#' @description
#' This `PipeOp` can be used to preprocess (one or more) [`lazy_tensor`] columns contained in an [`mlr3::Task`].
#' The preprocessing function is specified as construction argument `fn` and additional arguments to this
#' function can be defined through the `PipeOp`'s parameter set.
#' The preprocessing is done per column, i.e. the number of lazy tensor output columns is equal
#' to the number of lazy tensor input columns.
#'
#' To create custom preprocessing `PipeOp`s you can use [`pipeop_preproc_torch`] / [`pipeop_preproc_torch_class`].
#'
#' @section Inheriting:
#' In addition to specifying the construction arguments, you can overwrite the private `.shapes_out()` method.
#' If you don't overwrite it, the output shapes are assumed to be unknown (`NULL`).
#'
#' * `.shapes_out(shapes_in, param_vals, task)`\cr
#'   (`list()`, `list(), `Task` or `NULL`) -> `list()`\cr
#'   This private method calculates the output shapes of the lazy tensor columns that are created from applying
#'   the preprocessing function with the provided parameter values (`param_vals`).
#'   The `task` is very rarely needed, but if it is it should be checked that it is not `NULL`.
#'
#'   This private method only has the responsibility to calculate the output shapes for one input column, i.e. the
#'   input `shapes_in` can be assumed to have exactly one shape vector for which it must calculate the output shapes
#'   and return it as a `list()` of length 1.
#'   It can also be assumed that the shape is not `NULL` (i.e. unknown).
#'   Also, the first dimension can be `NA`, i.e. is unknown (as for the batch dimension).
#'
#' @template param_id
#' @template param_param_vals
#' @param fn (`function`)\cr
#'   The preprocessing function. Should not modify its input in-place.
#' @param packages (`character()`)\cr
#'   The packages the preprocessing function depends on.
#' @param param_set ([`ParamSet`])\cr
#'   In case the function `fn` takes additional parameter besides a [`torch_tensor`] they can be
#'   specfied as parameters. None of the parameters can have the [`"predict"`] tag.
#'   All tags should be set to `"train"`.
#' @param stages_init (`logical(1)`)\cr
#'   Initial value for the `stages` parameter.
#' @param rowwise (`logical(1)`)\cr
#'   Whether the preprocessing function is applied rowwise (and then concatenated by row) or directly to the whole
#'   tensor. In the first case there is no batch dimension.
#' @section Input and Output Channels:
#' See [`PipeOpTaskPreproc`].
#' @section State:
#' In addition to state elements from [`PipeOpTaskPreprocSimple`], the state also contains the `$param_vals` that
#' were set during training.
#'
#' @section Parameters:
#' In addition to the parameters inherited from [`PipeOpTaskPreproc`] as well as those specified during construction
#' as the argument `param_set` there are the following parameters:
#'
#' * `stages` :: `character(1)`\cr
#'   The stages during which to apply the preprocessing.
#'   Can be one of `"train"`, `"predict"` or `"both"`.
#'   The initial value of this parameter is set to `"train"` when the `PipeOp`'s id starts with `"augment_"` and
#'   to `"both"` otherwise.
#'   o
#'   Note that the preprocessing that is applied during `$predict()` uses the parameters that were set during
#'  `$train()` and not those that are set when performing the prediction.
#'
#' @section Internals:
#' During `$train()` / `$predict()`, a [`PipeOpModule`] with one input and one output channel is created.
#' The pipeop applies the function `fn` to the input tensor while additionally
#' passing the parameter values (minus `stages` and `affect_columns`) to `fn`.
#' The preprocessing graph of the lazy tensor columns is shallowly cloned and the `PipeOpModule` is added.
#' This is done to avoid modifying user input and means that identical `PipeOpModule`s can be part of different
#' preprocessing graphs. This is only possible, because the created `PipeOpModule` is stateless.
#'
#' At a later point in the graph, preprocessing graphs will be merged if possible to avoid unnecessary computation.
#' This is best illustrated by example:
#' One lazy tensor column's preprocessing graph is `A -> B`.
#' Then, two branches are created `B -> C` and `B -> D`, creating two preprocessing graphs
#' `A -> B -> C` and `A -> B -> D`. When loading the data, we want to run the preprocessing only once, i.e. we don't
#' want to run the `A -> B` part twice. For this reason, [`task_dataset()`] will try to merge graphs and cache
#' results from graphs.
#'
#' Also, the shapes created during `$train()` and `$predict()` might differ.
#' To avoid the creation of graphs where the predict shapes are incompatible with the train shapes,
#' the hypothetical predict shapes are already calculated during `$train()` (this is why the parameters that are set
#' during train are also used during predict) and the [`PipeOpTorchModel`] will check the train and predict shapes for
#' compatibility before starting the training.
#'
#' Otherwise, this mechanism is very similar to the [`ModelDescriptor`] construct.
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
#'   # get rid of environment baggage
#'   fn = mlr3misc::crate(function(x, a) x + a),
#'   param_set = ps(a = p_int(tags = c("train", "required")))
#' )
#'
#' po_simple$param_set$set_values(
#'   a = 100,
#'   affect_columns = selector_name(c("x1", "x2")),
#'   stages = "both" # use during train and predict
#' )
#'
#' taskout_train = po_simple$train(list(taskin))[[1L]]
#' materialize(taskout_train$data(cols = c("x1", "x2")), rbind = TRUE)
#'
#' taskout_predict_noaug = po_simple$predict(list(taskin))[[1L]]
#' materialize(taskout_predict_noaug$data(cols = c("x1", "x2")), rbind = TRUE)
#'
#' po_simple$param_set$set_values(
#'   stages = "train"
#' )
#'
#' # transformation is not applied
#' taskout_predict_aug = po_simple$predict(list(taskin))[[1L]]
#' materialize(taskout_predict_aug$data(cols = c("x1", "x2")), rbind = TRUE)
#'
#' # Creating a more complex preprocessing PipeOp
#'po_poly = pipeop_preproc_torch(
#'  public = list(
#'    initialize = function(id = "preproc_poly", param_vals = list()) {
#'      param_set = paradox::ps(
#'        n_degree = paradox::p_int(lower = 1L, tags = c("train", "required"))
#'      )
#'      param_set$set_values(
#'        n_degree = 1L
#'      )
#'      fn = mlr3misc::crate(function(x, n_degree) {
#'        torch::torch_cat(
#'          lapply(seq_len(n_degree), function(d) torch_pow(x, d)),
#'          dim = 2L
#'        )
#'      })
#'
#'      super$initialize(
#'        fn = fn,
#'        id = id,
#'        packages = character(0),
#'        param_vals = param_vals,
#'        param_set = param_set
#'      )
#'    }
#'  ),
#'  private = list(
#'    .shapes_out = function(shapes_in, param_vals, task) {
#'      # shapes_in is a list of length 1 containing the shapes
#'      checkmate::assert_true(length(shapes_in[[1L]]) == 2L)
#'      if (shapes_in[[1L]][2L] != 1L) {
#'        stop("Input shape must be (NA, 1)")
#'      }
#'      list(c(NA, param_vals$n_degree))
#'    }
#'  )
#')
#'
#'po_poly = PipeOpPreprocTorchPoly$new(
#'  param_vals = list(n_degree = 3L, affect_columns = selector_name("x3"))
#')
#'
#'po_poly$shapes_out(list(c(NA, 1L)))
#'
#'taskout = po_poly$train(list(tasken))[[1L]]
#'materialize(taskout$data(cols = "x3"), rbind = TRUE)
PipeOpTaskPreprocTorch = R6Class("PipeOpTaskPreprocTorch",
  inherit = PipeOpTaskPreproc,
  public = list(
    #' @description
    #' Creates a new instance of this [`R6`][R6::R6Class] class.
    initialize = function(fn, id = "preproc_torch", param_vals = list(), param_set = ps(), packages = character(0),
      stages_init = "both", rowwise = FALSE) { # nolint
      private$.fn = assert_function(fn)
      private$.rowwise = assert_flag(rowwise)

      param_set = assert_param_set(param_set$clone(deep = TRUE))

      param_set$add(ps(
        stages = p_fct(c("train", "predict", "both"), tags = c("train", "required"))
      ))
      param_set$set_values(stages = stages_init)

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
    #'  It uses the parameter values that are currently set.
    #' @param shapes_in (`list()` of (`integer()` or `NULL`))\cr
    #'   The input input shapes of the lazy tensors.
    #'   `NULL` indicates that the shape is unknown.
    #'   First dimension must be `NA`.
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

      stages = pv$stages
      if (identical(stages, "both")) stages = c("train", "predict")
      pv$stages = NULL
      pv$affect_columns = NULL

      if (stage %nin% stages) {
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
  active = list(
    #' @field fn
    #' The preprocessing function.
    fn = function(rhs) {
      assert_ro_binding(rhs)
      private$.fn
    },
    #' @field rowwise
    #' Whether the preprocessing is applied rowwise.
    rowwise = function(rhs) {
      assert_ro_binding(rhs)
      private$.rowwise
    }
  ),
  private = list(
    .rowwise = NULL,
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
      stages = param_vals$stages
      param_vals$stages = NULL
      trafo = private$.fn

      fn = if (identical(stages, "both") || stage %in% stages) {
        if (length(param_vals)) {
          if (private$.rowwise) {
            trafo = crate(function(x) {
              torch_cat(map(seq_len(nrow(x)), function(i) invoke(trafo, x[i, ..], .args = param_vals)$unsqueeze(1L)), dim = 1L)
            }, param_vals, trafo, .parent = environment(trafo))
          } else {
            crate(function(x) {
              invoke(.f = trafo, x, .args = param_vals)
            }, param_vals, trafo, .parent = environment(trafo))
          }
        } else {
          if (private$.rowwise) {
            crate(function(x) {
              torch_cat(map(seq_len(nrow(x)), function(i) trafo(x[i, ..])$unsqueeze(1L)), dim = 1L)
            }, trafo)
          } else {
            trafo
          }
        }
      } else {
        identity
      }

      dt = imap_dtc(dt, function(lt, nm) {
        po_fn = PipeOpModule$new(
          id = paste0(self$id, ".", nm),
          module = fn,
          inname = self$input$name,
          outname = self$output$name,
          packages = self$packages
        )

        shape_before = dd(lt)$pointer_shape
        shape_out = self$shapes_out(list(shape_before), stage = stage, task = task)[[1L]]

        shape_out_predict = if (stage == "train") {
          shape_in_predict = if (is.null(dd(lt)$pointer_shape_predict)) shape_before
          # during `$train()` we also keep track of the shapes that would arise during predict
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
      # no need to ensure uniqueness of names as we are just taking the names as they already are
      return(dt)
    },
    .additional_phash_input = function() {
      list(
        self$param_set$ids(), self$packages,
        formals(private$.fn), body(private$.fn), address(environment(private$.fn))
      )
    },
    .shapes_out = function(shapes_in, param_vals, task) list(NULL),
    .fn = NULL
  )
)
#' @title Create Torch Preprocessing PipeOps
#' @description
#' Calls [`pipeop_preproc_torch_class`] and instantiates the instance with the given parameter values.
#' @inheritParams pipeop_preproc_torch_class
#' @template param_param_vals
#' @param param_vals (`list()`)\cr
#'   The parameter values.
#' @export
pipeop_preproc_torch = function(id, fn, shapes_out, param_set = NULL, param_vals = list(), packages = character(0),
  rowwise = FALSE, parent_env = parent.frame()) {
  pipeop_preproc_torch_class(
    id = id,
    fn = fn,
    shapes_out = shapes_out,
    param_set = param_set,
    packages = packages,
    rowwise = rowwise,
    parent_env = parent_env
    )$new(param_vals = param_vals)
}


create_ps = function(fn) {
  # TODO: could simplify this as we don't need the expression anymore
  fmls = rlang::fn_fmls(fn)
  param_names = names(fmls)
  param_names = setdiff(param_names[-1L], "...")
  fmls = fmls[param_names]
  is_required = map(fmls, function(x) identical(x, rlang::missing_arg()))
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

  eval(result)
}

#' @title Create Torch Preprocessing PipeOps
#' @description
#' Function to create objects of class [`PipeOpTaskPreprocTorch`] in a slightly more convenient way.
#' Start by reading the documentation of [`PipeOpTaskPreprocTorch`].
#' @template param_id
#' @param fn (`function`)\cr
#'   The preprocessing function.
#' @param shapes_out (`function` or `NULL` or `"infer"` or `"unchanged"`)\cr
#'   The private `.shapes_out(shapes_in, param_vals, task)` method of [`PipeOpTaskPreprocTorch`]
#'   (see section Inheriting).
#'   Special values are `NULL` and `infer`:
#'   If `NULL`, the output shapes are unknown.
#'   If "infer", the output shape function is inferred and calculates the output shapes as follows:
#'   For an input shape of (NA, ...) a meta-tensor of shape (1, ...) is created and the preprocessing function is
#'   applied. Afterwards the batch dimension (1) is replaced with NA and the shape is returned.
#'
#'   This should be correct in most cases, but might fail in some edge cases.
#' @param param_set ([`ParamSet`] or `NULL`)\cr
#'   The parameter set.
#'   If this is left as `NULL` (default) the parameter set is inferred in the following way:
#'   All parameters but the first and `...` of `fn` are set as untyped parameters with tags 'train' and those that
#'   have nod default value are tagged as 'required' as well.
#'   Default values are not annotated.
#' @template param_packages
#' @export
#' @returns An [`R6Class`][R6::R6Class] instance inheriting from [`PipeOpTaskPreprocTorch`]
#' @examples
#' po_example = pipeop_preproc_torch("preproc_example", function(x, a) x + a)
#' po_example
#' po_example$param_setents cannot be passed as variables but need to be passed as expressions.
pipeop_preproc_torch_class = function(id, fn, shapes_out, param_set = NULL, packages = character(0),
  init_params = list(), rowwise = FALSE, parent_env = parent.frame(), leanify = NULL) {
  assert(
    check_function(shapes_out, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE),
    check_choice(shapes_out, c("infer", "unchanged"))
  )
  if (identical(shapes_out, "infer")) {
    shapes_out = crate(function(shapes_in, param_vals, task) {
      sin = shapes_in[[1L]]
      batch_dim = sin[1L]
      batchdim_is_unknown = is.na(batch_dim)
      if (batchdim_is_unknown) {
        sin[1] = 1L
      }
      if (self$rowwise) {
        sin = sin[-1L]
      }
      tensor_in = invoke(torch_empty, .args = sin, device = torch_device("meta"))
      tensor_out = tryCatch(invoke(private$.fn, tensor_in, .args = param_vals),
        error = function(e) {
          stopf("Failed to infer output shape, presumably invalid input shape; error message is: %s", e)
        }
      )
      sout = dim(tensor_out)

      if (self$rowwise) {
        sout = c(batch_dim, sout)
      } else if (batchdim_is_unknown) {
        sout[1] = NA
      }

      list(sout)
    })
    # FIXME: NSE issues, we need to evaluate fn in the proper environment,
    # I guess we can use this quosure idea that
  } else if (identical(shapes_out, "unchanged")) {
    shapes_out = crate(function(shapes_in, param_vals, task_in) shapes_in)
  } else if (is.function(shapes_out) || is.null(shapes_out)) {
    # nothing to do
  } else {
    stopf("unreachable")
  }

  param_set = param_set %??% create_ps(fn)

  stages_init = if (startsWith(id, "augment_")) "train" else "both"
  init_fun = crate(function(id = idname, param_vals = list()) { # nolint
    info = private$.__construction_info
    param_set = info$param_set$clone(deep = TRUE)
    param_set$values = info$init_params # nolint
    super$initialize(
      id = id,
      packages = info$packages,
      param_set = param_set,
      param_vals = param_vals,
      fn = info$fn, # nolint
      rowwise = info$rowwise,
      stages_init = info$stages_init
    )
    # no need to keep the data around after initialiation
    private$.__construction_info = NULL
  }, param_set, stages_init, rowwise, packages, init_params)

  formals(init_fun)$id = id

  classname = paste0("PipeOpPreprocTorch", paste0(capitalize(strsplit(id, split = "_")[[1L]]), collapse = ""))
  # Note that we don't set default values

  private = list(
    .__construction_info = list(
      packages = packages,
      param_set = param_set,
      fn = fn,
      rowwise = rowwise,
      stages_init = stages_init,
      init_params = init_params
    )
  )

  if (!is.null(shapes_out)) {
    private$.shapes_out = shapes_out
  }

  Class = R6Class(classname,
    inherit = PipeOpTaskPreprocTorch,
    public = list(
      initialize = init_fun
    ),
    private = private,
    parent_env = parent_env
  )

  return(Class)
}

register_preproc = function(id, fn, param_set = NULL, shapes_out = NULL, packages = character(0), rowwise = FALSE) {
  Class = pipeop_preproc_torch_class(id, fn, param_set = param_set, shapes_out = shapes_out,
    packages = packages, rowwise = rowwise, parent_env = parent.frame())
  assign(Class$classname, Class, parent.frame())
  register_po(id, Class)
}

register_po("preproc_torch", PipeOpTaskPreprocTorch, metainf = list(fn = identity, rowwise = FALSE))
