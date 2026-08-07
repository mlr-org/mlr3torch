#' @title Create a Torch PipeOp
#'
#' @description
#' Creates a class inheriting from [`PipeOpTorch`] that wraps an [`nn_module`][torch::nn_module],
#' without having to write the R6 class by hand.
#' Start by reading the *Inheriting* section of [`PipeOpTorch`], which describes what the generated
#' class does; this function is a shortcut for the common cases.
#'
#' The two things a [`PipeOpTorch`] has to know beyond the module itself are handled by the
#' `auxiliary` and `shapes_out` arguments:
#' * arguments of the module that are not set by the user but follow from the shape of the input,
#'   such as `in_features` of [`nn_linear`][torch::nn_linear], and
#' * the shape of the tensors that the module produces.
#'
#' @template param_id
#' @param module_generator (`nn_module_generator`)\cr
#'   The module that the `PipeOp` wraps, e.g. [`nn_linear`][torch::nn_linear].
#' @param param_set ([`ParamSet`][paradox::ParamSet] or `NULL`)\cr
#'   The parameter set.
#'   If left as `NULL` (default), it is inferred from the arguments of the module's `$initialize()`
#'   method: each becomes an untyped parameter tagged `"train"`, except for the names of
#'   `auxiliary`, which the user does not set.
#' @param auxiliary (named `list()` of `function`s)\cr
#'   The arguments of the module that are inferred from the input shapes rather than set by the
#'   user, e.g. `in_features` of [`nn_linear`][torch::nn_linear].
#'   Each element is a `function(shapes_in, param_vals, task)` returning the value for the argument
#'   it is named after; `shapes_in` is named after the input channels.
#'   Implemented as the private `$.shape_dependent_params()` method of [`PipeOpTorch`], so pass
#'   `shape_dependent_params` instead if the arguments cannot be computed one at a time.
#' @param shape_dependent_params (`function` or `NULL`)\cr
#'   The private `$.shape_dependent_params(shapes_in, param_vals, task)` method of [`PipeOpTorch`],
#'   for the cases `auxiliary` cannot express.
#'   It must return *all* arguments that are passed to the module, i.e. `param_vals` plus the
#'   inferred ones. Cannot be combined with `auxiliary`.
#' @param shapes_out (`function` or `"infer"` or `NULL`)\cr
#'   The private `$.shapes_out(shapes_in, param_vals, task)` method of [`PipeOpTorch`].
#'   With the default `"infer"`, the module is built and traced on the "meta" device, i.e. without
#'   allocating any memory, which is correct as long as the output shape does not depend on the
#'   *values* in the tensor. `NULL` keeps the shapes unchanged, which is what shape-preserving
#'   operators such as activation functions want.
#' @param inname (`character()` or `NULL`)\cr
#'   The names of the input channels. If `NULL` (default), the argument names of the module's
#'   `$forward()` method are used.
#' @param outname (`character()`)\cr
#'   The names of the output channels, `"output"` by default.
#'   A module with more than one output channel must return a named `list()`.
#' @param classname (`character(1)`)\cr
#'   The class name of the generated [`R6Class`][R6::R6Class].
#' @param parent_env (`environment`)\cr
#'   The parent environment for the R6 class.
#' @template param_packages
#' @param tags (`character()`)\cr
#'   Tags for the `PipeOp`. The tag `"torch"` is always added.
#'
#' @return An [`R6Class`][R6::R6Class] generator inheriting from [`PipeOpTorch`].
#'
#' @family Graph Network
#' @family PipeOps
#' @export
#' @examplesIf torch::torch_is_installed()
#' # A layer that scales its input by a learned factor, with the number of features -- which the
#' # user cannot know when the network is built -- inferred from the input shape.
#' nn_scale = nn_module("nn_scale",
#'   initialize = function(n_features, init = 1) {
#'     self$weight = nn_parameter(torch_full(n_features, init))
#'   },
#'   forward = function(input) input * self$weight
#' )
#'
#' PipeOpTorchScale = pipeop_torch("nn_scale", nn_scale,
#'   auxiliary = list(
#'     n_features = function(shapes_in, param_vals, task) tail(shapes_in[[1L]], 1L)
#'   )
#' )
#'
#' po_scale = PipeOpTorchScale$new()
#' # `init` is a hyperparameter, `n_features` is not
#' po_scale$param_set$ids()
#' po_scale$shapes_out(list(c(NA, 4)))
#'
#' # the operator can now be used like any other, and `nn_scale` is built with n_features = 4
#' md = po("torch_ingress_num") %>>% po_scale %>>% po("nn_head")
#' network = model_descriptor_to_module(md$train(tsk("iris"))[[1L]])
#' network
pipeop_torch = function(id, module_generator, param_set = NULL, auxiliary = NULL,
  shape_dependent_params = NULL, shapes_out = "infer", inname = NULL, outname = "output",
  packages = character(0), tags = NULL, classname = NULL, parent_env = parent.frame()) {
  assert_string(id)
  assert_class(module_generator, "nn_module_generator")
  if (!is.null(param_set)) assert_param_set(param_set)
  assert_list(auxiliary, types = "function", names = "unique", null.ok = TRUE)
  assert_function(shape_dependent_params, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE)
  assert(
    check_function(shapes_out, args = c("shapes_in", "param_vals", "task"), null.ok = TRUE),
    check_choice(shapes_out, "infer")
  )
  assert_character(inname, null.ok = TRUE)
  assert_character(outname, min.len = 1L)
  assert_character(packages, any.missing = FALSE)
  assert_character(tags, null.ok = TRUE)
  assert_string(classname, null.ok = TRUE)
  if (!is.null(auxiliary) && !is.null(shape_dependent_params)) {
    stopf("Pass either 'auxiliary' or 'shape_dependent_params', not both.")
  }

  # a module that only implements `$forward()` has no arguments to infer anything from
  init = get_init(module_generator)
  init_args = if (is.null(init)) character(0) else names(formals(init))
  if (!is.null(auxiliary)) {
    assert_subset(names(auxiliary), init_args, .var.name = "names of 'auxiliary'")
  }
  if (!is.null(param_set)) {
    assert_disjunct(param_set$ids(), names(auxiliary),
      .var.name = "parameter ids and the names of 'auxiliary'")
  }

  classname = classname %??% paste0("PipeOpTorch",
    paste0(capitalize(strsplit(sub("^nn_", "", id), split = "_")[[1L]]), collapse = ""))

  # the parameter set is built during construction so that each instance owns its own copy
  init_fun = crate(function(id = id, param_vals = list()) { # nolint
    info = private$.__construction_info
    super$initialize(
      id = id,
      module_generator = info$module_generator,
      param_set = info$param_set %??% inferps(info$module_generator, ignore = names(info$auxiliary) %??% character(0)),
      param_vals = param_vals,
      inname = info$inname %??% setdiff(names(formals(info$module_generator$public_methods$forward)), "..."),
      outname = info$outname,
      packages = info$packages,
      tags = info$tags
    )
    private$.__construction_info = NULL
  }, .parent = topenv())
  formals(init_fun)$id = id

  private = list(
    .__construction_info = list(
      module_generator = module_generator,
      param_set = if (!is.null(param_set)) param_set$clone(deep = TRUE),
      auxiliary = auxiliary,
      inname = inname,
      outname = outname,
      packages = packages,
      tags = tags
    ),
    .auxiliary = auxiliary
  )

  if (!is.null(auxiliary)) {
    private$.shape_dependent_params = crate(function(shapes_in, param_vals, task) {
      c(param_vals, lapply(private$.auxiliary, function(fn) fn(shapes_in, param_vals, task)))
    }, .parent = topenv())
  } else if (!is.null(shape_dependent_params)) {
    private$.shape_dependent_params = shape_dependent_params
  }

  if (identical(shapes_out, "infer")) {
    private$.shapes_out = crate(function(shapes_in, param_vals, task) {
      getFromNamespace("infer_shapes_module", "mlr3torch")(shapes_in = shapes_in,
        make_module = function() private$.make_module(shapes_in, param_vals, task), # nolint
        input_names = self$input$name, output_names = self$output$name, id = self$id) # nolint
    }, .parent = topenv())
  } else if (is.function(shapes_out)) {
    private$.shapes_out = shapes_out
  }

  R6Class(classname,
    inherit = PipeOpTorch,
    public = list(initialize = init_fun),
    private = private,
    parent_env = parent_env
  )
}

# The module counterpart of `infer_shapes()`: the output shapes of a module are read off tensors
# that are pushed through it, rather than being computed. As there, the unknown dimensions are
# filled in more than once and a dimension that does not come out the same every time is unknown.
# Unlike `infer_shapes()`, all inputs are traced together, because a module's inputs are not
# independent of one another.
infer_shapes_module = function(shapes_in, make_module, input_names, output_names, id) {
  assert_shapes(shapes_in)
  # The module is rebuilt per trace because it is moved to the device that is traced on, and a
  # module on the "meta" device cannot be moved back.
  trace = function(na_repl, device) {
    module = make_module()
    module$to(device = torch_device(device))
    tensors = lapply(shapes_in, function(shape) {
      shape[is.na(shape)] = na_repl
      mlr3misc::invoke(torch_empty, .args = as.list(as.integer(shape)), device = torch_device(device))
    })
    names(tensors) = input_names[seq_along(tensors)]
    out = with_no_grad(mlr3misc::invoke(module$forward, .args = tensors))
    if (inherits(out, "torch_tensor")) out = list(out)
    lapply(out, dim)
  }

  traced = lapply(na_replacements(unlist(shapes_in)), function(na_repl) {
    # "meta" allocates nothing, so the trace costs no memory however large the shape is, but not
    # every operator implements it; one that does not raises rather than returning a wrong shape.
    tryCatch(list(shapes = trace(na_repl, "meta")), error = function(e) {
      tryCatch(list(shapes = trace(na_repl, "cpu")), error = function(e) list(condition = e))
    })
  })
  shapes = map(Filter(function(x) !is.null(x$shapes), traced), "shapes")
  if (length(shapes) < 2L) {
    # the traces with the larger values are the informative ones, so report the last failure
    condition = last(Filter(function(x) !is.null(x$condition), traced))$condition
    stopf("%s\nThe output shapes of PipeOp with id '%s' could not be inferred by tracing the module, specify them explicitly instead (see the `shapes_out` argument).", # nolint
      conditionMessage(condition), id)
  }
  if (length(unique(lengths(shapes))) > 1L) {
    stopf("Failed to infer shapes for PipeOp with id '%s', as the number of outputs varies with the values filled in for the unknown dimensions.", id) # nolint
  }

  # one shape per output channel, unknown wherever the traces disagree
  shapes_out = lapply(seq_along(shapes[[1L]]), function(i) {
    per_trace = map(shapes, i)
    if (length(unique(lengths(per_trace))) > 1L) {
      stopf("Failed to infer shapes for PipeOp with id '%s', as the number of dimensions varies with the values filled in for the unknown dimensions.", id) # nolint
    }
    as.integer(apply(do.call(rbind, per_trace), 2L, function(xs) if (length(unique(xs)) == 1L) xs[[1L]] else NA))
  })

  if (length(shapes_out) != length(output_names)) {
    stopf("PipeOp with id '%s' has %i output channel(s), but its module returned %i output(s).",
      id, length(output_names), length(shapes_out))
  }
  set_names(shapes_out, output_names)
}

#' @title Convert a Torch Module to a PipeOp
#'
#' @description
#' Wraps an [`nn_module`][torch::nn_module] into a [`PipeOpTorch`], so that it can be used in a
#' [`Graph`][mlr3pipelines::Graph].
#' All arguments of the module's `$initialize()` become hyperparameters of the `PipeOp`, so a module
#' with arguments that have to be inferred from the input shapes -- such as `in_features` of
#' [`nn_linear`][torch::nn_linear] -- needs [`pipeop_torch()`] and its `auxiliary` argument instead.
#'
#' The `PipeOp`'s id is the module's class name. Use [`pipeop_torch()`] directly to choose another
#' one, or to configure anything else about the resulting class.
#'
#' @param x (`nn_module_generator`)\cr
#'   The module to wrap.
#' @param clone (`logical(1)`)\cr
#'   Ignored, a new [`PipeOpTorch`] is created in either case.
#' @return A [`PipeOpTorch`].
#' @export
#' @examplesIf torch::torch_is_installed()
#' nn_square = nn_module("nn_square", forward = function(input) input^2)
#' po_square = as_pipeop(nn_square)
#' po_square$id
#' po_square$shapes_out(list(c(NA, 4)))
as_pipeop.nn_module_generator = function(x, clone = FALSE) { # nolint
  id = assert_string(x$classname, .var.name = "the module's class name")
  pipeop_torch(id = id, module_generator = x, parent_env = parent.frame())$new()
}
