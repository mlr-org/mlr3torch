#' @title Base Class for Torch Preprocessing
#' @name mlr_pipeops_preproc_torch
#'
#' @description
#' This `PipeOp` can be used to preprocess (one or more) [`lazy_tensor`] columns contained in an [`mlr3::Task`].
#' The function that is applied is specified as construction argument `fn` and additional arguments to this
#' function can be defined through the `PipeOp`'s parameter set.
#' The preprocessing is either done per-column in which case the number lazy tensor output columns is equivalent
#' to the number of lazy tensor input columns.
#' It is also possible to implement preprocessing that is applied to all lazy tensor columns at once and returns
#' one or more (not necessarily the same number) of lazy tensor columns.
#'
#' @section Inheriting:
#' In addition to specifying the parameters `fn`, `packages` and `param_set` during construction you can also overwrite
#' the private `.shapes_out()` or `.tranform()` methods:
#' * `.shapes_out(shapes_in, param_vals, task)`\cr
#'   (`list()`, `list(), `Task` or `NULL`) -> `list()`\cr
#'   This private method calculates the output shapes of the lazy tensor columns that are created from applying
#'   the preprocessing.
#'
#'   Also see the documentation of [`PipeOpTorch`] how to implement this method.
#'
#'   In case the construction argument `per_column` is `TRUE`, this private method only has the responsibility
#'   to caclculate the output shapes for one input column, i.e. the input `shapes_in` can be assumed to have
#'   exactly one shape vector for which it must calculate the output shapes.
#'
#' * `.transform(dt, task, param_vals, stage)`\cr
#'   (`data.table()`, `Task`, `list()`, `character(1)`) -> `data.table()`\cr
#'   This method must only be overwritten when the the `per_column` construction argument is `FALSE`.
#'   It receives as inputs all selected lazy tensor columns, the input `Task` (already cloned),
#'   the paramer values, and whether the preprocessing is applied during training (stage is `"train"`)
#'   or prediction (stage is `"predict"`). It needs to return a `data.table` with lazy tensor columns.
#'   Note that the lazy tensor inputs should not be modified in-place.
#'   Note that overwriting this method (currently) requires a solid understanding of the [`lazy_tensor`] internals. This might be made easier in the future.
#'   Note also that you need to pay attention to avoid name conflicts with existing columns in the task.
#'
#' @template param_id
#' @template param_param_vals
#' @param fn (`function`)\cr
#'   The preprocessing function.
#' @param packages (`character()`)\cr
#'   The packages the preprocessing function depends on.
#' @param param_set ([`ParamSet`])\cr
#'   In case the function `fn` takes additional parameter besides a [`torch_tensor()`] they can be
#'   specfied as parameters. Pay attention to set the correct `tags` for the parameters: if tag `"train"` is present,
#'   the preprocessing is applied during training and if tag `"predict"` is present, the preprocessing is applied
#'   during prediction (if `augment` is set to `FALSE`).
#' @param per_column (`logical(1)`)\cr
#'   Whether the transformation is applied per column.
#'   If this is `FALSE`, is applied to all lazy tensor columns at once and might produce
#'   one or more new lazy tensor columns.
#' @param augment_init (`logical(1)`)\cr
#'   Initial value for the `augment` parameter.
#'
#' @section Input and Output Channels:
#' See [`PipeOpTaskPreproc`].
#' @section State:
#' See [`PipeOpTaskPreproc`].
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
#' If `per_column` is `TRUE`:
#'
#' A [`PipeOpModule`] with one input and one output channel is created.
#' The pipeop simply applies the function `fn` to the input tensor while additionally
#' passing the paramter values (minus `augment` and `affect_columns`) to `fn`.
#' Then [`transform_lazy_tensor`] is called with the created [`PipeOpModule`] and the shapes obtained from the
#' `$shapes_out()` method of this `PipeOp`.
#'
#'
#' If `per_column` is `FALSE`:
#' It is the obligation of the user to overwrite the `.transform()` method appropriately.
#' Note that on
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
#' materialize(taskout_train$data(cols = c("x1", "x2")))
#'
#' taskout_predict_noaug = po_simple$predict(list(taskin))[[1L]]
#' materialize(taskout_predict_noaug$data(cols = c("x1", "x2")))
#'
#' po_simple$param_set$set_values(
#'   augment = TRUE
#' )
#'
#' # transformation is not applied
#' taskout_predict_aug = po_simple$predict(list(taskin))[[1L]]
#' materialize(taskout_predict_aug$data(cols = c("x1", "x2")))
#'
#' # Creating a more complex preprocessing PipeOp
#'
#' PipeOpPreprocTorchPoly = R6::R6Class("PipeOpPreprocTorchPoly",
#'   inherit = PipeOpTaskPreprocTorch,
#'   public = list(
#'     initialize = function(id = "preproc_poly", param_vals = list()) {
#'       param_set = paradox::ps(
#'         n_degree = paradox::p_int(lower = 1L, tags = c("train", "predict", "required"))
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
#' # Note that the 'augment' parameter is not present as the PipeOp
#' # modifies the input shape and must hence be applied during training **and** prediction
#' po_poly$param_set
#'
#' po_poly$shapes_out(list(c(NA, 1L)))
#'
#' taskout = po_poly$train(list(taskin))[[1L]]
#' materialize(taskout$data(cols = "x3"))
PipeOpTaskPreprocTorch = R6Class("PipeOpTaskPreprocTorch",
  inherit = PipeOpTaskPreproc,
  public = list(
    #' @description
    #' Creates a new instance of this [`R6`][R6::R6Class] class.
    initialize = function(fn, id = "preproc_torch", param_vals = list(), param_set = ps(), packages = character(0),
      per_column = TRUE, augment_init = FALSE) {
      private$.per_column = assert_flag(per_column)
      private$.fn = assert_function(fn, null.ok = per_column)
      assert_flag(augment_init)

      if (per_column) {
        param_set$add(ps(
          augment = p_lgl(tags = c("predict", "required"))
        ))
        param_set$set_values(augment = augment_init)
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
    #' @param shapes_in (`list()` of `integer()`)\cr
    #'   The input input shapes of the lazy tensors.
    #' @param task ([`Task`] or `NULL`)\cr
    #'   The task, which is very rarely needed.
    #' @param stage (`character(1)`)\cr
    #'   The stage: either `"train"` or `"predict"`.
    #' @return `list()` of `integer()`
    shapes_out = function(shapes_in, stage = NULL, task = NULL) {
      assert_r6(task, "Task", null.ok = TRUE)
      assert_shapes(shapes_in, named = FALSE)
      names(shapes_in) = NULL

      if (is.null(private$.shapes_out)) {
        # shapes_out can only be NULL if per_column is TRUE
        return(shapes_in)
      }

      assert_choice(stage, c("train", "predict"))

      pv = self$param_set$get_values(tags = stage)

      s = if (private$.per_column) {
        map(shapes_in, function(s) private$.shapes_out(list(s), param_vals = pv, task = task)[[1L]])
      } else {
        private$.shapes_out(shapes_in, param_vals = pv, task = task)
      }

      return(s)
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
      dt = private$.transform(dt, task, param_vals, "train")

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
      dt = private$.transform(dt, task, param_vals, "predict")
      res = task$select(setdiff(task$feature_names, cols))$cbind(dt)

      res
    },
    .transform = function(dt, task, param_vals, stage) {
      # stage is "train" or "predict"
      trafo = private$.fn

      if (stage == "train") {
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
        x = transform_lazy_tensor(lt, po_fn, shape_out)
        dd = x$data_descriptor

        # During "train" we will also calculate the shapes that would arise during prediction, making
        # some assumption that the task and the parameter values are the same.
        # This is then later used in PipeOpTorchModel **before** starting the training to verify that
        # a `$predict()` call is possible later.
        if (stage == "train") {
          shape_out_predict = self$shapes_out(list(shape_before), stage = "predict", task = task)[[1L]]
          dd$.info$.pointer_shape_predict = shape_out_predict
          attr(x, "data_descriptor") = dd
        }
        x
      })
      names(dt) = uniqueify(names(dt), setdiff(task$col_info$id, names(dt)))
      return(dt)
    },
    .additional_phash_input = function() {
      list(self$param_set$ids(), private$.fn, self$packages, private$.per_column)
    },
    .fn = NULL,
    .per_column = NULL
  )
)

#' @title Create Torch Preprocessing PipeOps
#' @description
#' Calls [`pipeop_preproc_torch_class`] and instantiates the instance with the given parameter values.
#' @inheritParams pipeop_preproc_torch_class
#' @param param_vals (`list()`)\cr
#'   The parameter values.
#' @export
pipeop_preproc_torch = function(name, fn, shapes_out = NULL, param_set = NULL, param_vals = list(),
  packages = character(0), per_column = TRUE, prefix = "trafo") {
  pipeop_preproc_torch_class(
    name = name,
    fn = fn,
    shapes_out = shapes_out,
    param_set = param_set,
    packages = packages,
    per_column = per_column,
    prefix = prefix
    )$new(param_vals = param_vals)
}


create_ps_call = function(v) {
  # Create an empty named list to store the arguments
  args = list()

  # Iterate through the elements of v and create expressions
  for (element in v) {
    arg_name = as.name(element)
    arg_value = rlang::expr(p_uty(tags = c("train", "predict")))
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
#' @param shapes_out (`function` or `NULL` or `TRUE`)\cr
#'   The private `.shapes_out(shapes_in, param_vals, task)` method of [`PipeOpTaskPreprocTorch`].
#'   If `NULL`, the pipeop does not change the output shapes.
#'   If `TRUE`, the output shape function is inferred and calculates the output shapes as follows:
#'   For an input shape of (NA, ...) a meta-tensor of shape (1, ...) is created and the preprocessing function is
#'   applied. Afterwards the batch dimension (1) is replaced with NA and the shape is returned.
#' @param param_set ([`ParamSet`] or `NULL`)\cr
#'   The parameter set.
#'   If this is left as `NULL` (default) the parameter set is inferred in the following way:
#'   All arguments but the first and `...` of `fn` are set as untyped parameters with tags 'train' and 'predict'.
#'   Default values are not annotated.
#' @template param_param_vals
#' @template param_packages
#' @param per_column (`logical(1)`)\cr
#'   Whether the preprocessing is applied per-column.
#' @export
#' @returns An [`R6Class`][R6::R6Class] instance inheriting from [`PipeOpTaskPreprocTorch`]
#' @examples
#' po_example = pipeop_preproc_torch("preproc_example", function(x, a) x + a)
#' po_example
#' po_example$param_set
pipeop_preproc_torch_class = function(name, fn, shapes_out = NULL, param_set = NULL,
  packages = character(0), per_column = TRUE, prefix = "trafo") {
  assert_string(name)
  assert_function(fn)
  if (!isTRUE(shapes_out)) {
    assert_function(shapes_out, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE)
  }
  assert_character(packages)
  assert_flag(per_column)

  if (!is.null(param_set)) {
    assert_param_set(param_set)
    param_set = substitute(param_set)
  } else {
    param_names = setdiff(formalArgs(fn)[-1], "...")
    param_set = create_ps_call(param_names)
  }

  classname = paste0("PipeOpPreprocTorch", paste0(capitalize(strsplit(name, split = "_")[[1L]]), collapse = ""))
  # Note that we don't set default values


  if (isTRUE(shapes_out)) {
    shapes_out = crate(function(shapes_in, param_vals, task) {
      sin = shapes_in[[1L]]
      # set batch-dim to 1
      sin[1] = 1L
      tensor_in = invoke(torch_empty, .args = sin, device = torch_device("meta"))
      tensor_out = invoke(private$.fn, tensor_in, .args = param_vals)

      sout = dim(tensor_out)
      sout[1] = NA
      list(sout)
    })
  }

  idname = paste0(prefix, "_", name)

  init_fun = crate(function(id = idname, param_vals = list()) {
    super$initialize(
      id = id,
      packages = packages,
      param_set = ps,
      param_vals = param_vals,
      fn = fn
    )
  })
  formals(init_fun)$id = idname
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

register_preproc = function(name, fn, param_set = NULL, shapes_out = NULL, packages, prefix = "trafo") {
  id = register_po(paste0(prefix, "_", name), Class)
  class = pipeop_preproc_torch_class(id, fn, param_set = param_set, shapes_out = shapes_out, packages = packages)

  register_po(id, class)

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


register_po("preproc_torch", PipeOpTaskPreprocTorch)
