#' @title Create a Torch PipeOp
#'
#' @description
#' Creates a class inheriting from [`PipeOpTorch`] that wraps an [`nn_module`][torch::nn_module],
#' without having to write either of them by hand.
#' Start by reading the *Inheriting* section of [`PipeOpTorch`], which describes what the generated
#' class does; this function is a shortcut for the common cases.
#'
#' The module is written as it would be with [`nn_module()`][torch::nn_module], except that
#' `initialize` can take two further arguments that the `PipeOp` supplies: `shapes_in` and `task`.
#' This is what makes a layer whose size follows from the data -- the `in_features` of a
#' [`nn_linear`][torch::nn_linear], say -- possible without the user having to state it.
#'
#' A module that states its own output shapes needs none of this and is converted by
#' [`as_pipeop()`][mlr3pipelines::as_pipeop] instead.
#'
#' @template param_id
#' @param initialize (`function` or `NULL`)\cr
#'   The `$initialize()` method of the module.
#'   Its arguments become the hyperparameters of the `PipeOp`, except for `shapes_in` (the input
#'   shapes, named after the input channels) and `task` (the [`Task`][mlr3::Task] or `NULL`), which
#'   are supplied by the `PipeOp` and passed only if the function declares them.
#' @param forward (`function`)\cr
#'   The `$forward()` method of the module.
#' @param shapes_out (`function`)\cr
#'   The shapes of the tensors that the module produces, as a function of `shapes_in`, `param_vals`
#'   and `task` -- only the arguments that are declared are passed, so an operator whose output
#'   shape depends on nothing else is written as `function(shapes_in)`, and one that does not change
#'   the shape as `function(shapes_in) shapes_in`.
#'   It returns one shape per output channel, in the order of `out_channels`.
#'   Any dimension of `shapes_in` can be `NA`, i.e. unknown, so this function must not assume that a
#'   dimension it reads is known; see the *Inheriting* section of [`PipeOpTorch`].
#' @param param_set ([`ParamSet`][paradox::ParamSet] or `NULL`)\cr
#'   The parameter set.
#'   If left as `NULL` (default), it is inferred from the arguments of `initialize`: each becomes an
#'   untyped parameter tagged `"train"`, and additionally `"required"` if it has no default.
#'   The arguments that the `PipeOp` supplies -- `shapes_in` and `task` -- are not among them.
#' @param in_channels (`character()` or `integer(1)` or `NULL`)\cr
#'   The input channels, either as names or as a count, where `0` means a single *vararg* channel.
#'   If `NULL` (default), the arguments of `forward` are used.
#' @param out_channels (`character()` or `integer(1)`)\cr
#'   The output channels, either as names or as a count, `1` by default.
#'   A module with more than one output channel must return a `list()`, in the order of the
#'   channels.
#' @param classname (`character(1)`)\cr
#'   The class name of the generated [`R6Class`][R6::R6Class].
#' @param parent_env (`environment`)\cr
#'   The environment in which the module's methods are evaluated, the calling environment by
#'   default, as for [`nn_module()`][torch::nn_module].
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
#' # user cannot know when the network is built -- read off the input shape.
#' PipeOpTorchScale = pipeop_torch("nn_scale",
#'   initialize = function(shapes_in, init = 1) {
#'     self$weight = nn_parameter(torch_full(tail(shapes_in[[1L]], 1L), init))
#'   },
#'   forward = function(input) input * self$weight,
#'   shapes_out = function(shapes_in) shapes_in # scaling leaves the shape as it is
#' )
#'
#' po_scale = PipeOpTorchScale$new()
#' # `init` is a hyperparameter, the number of features is not
#' po_scale$param_set$ids()
#' po_scale$shapes_out(list(c(NA, 4)))
#'
#' # the operator can now be used like any other, and the module is built with 4 features
#' md = po("torch_ingress_num") %>>% po_scale %>>% po("nn_head")
#' network = model_descriptor_to_module(md$train(tsk("iris"))[[1L]])
#' network
#'
#' # A module that already exists is wrapped by calling it from `initialize()`
#' PipeOpTorchLinear2 = pipeop_torch("nn_linear2",
#'   initialize = function(shapes_in, out_features, bias = TRUE) {
#'     self$linear = nn_linear(tail(shapes_in[[1L]], 1L), out_features, bias)
#'   },
#'   forward = function(input) self$linear(input),
#'   shapes_out = function(shapes_in, param_vals) {
#'     list(c(head(shapes_in[[1L]], -1L), param_vals$out_features))
#'   }
#' )
#' PipeOpTorchLinear2$new(param_vals = list(out_features = 10))$shapes_out(list(c(NA, 4)))
pipeop_torch = function(id, initialize = NULL, forward, shapes_out, param_set = NULL,
  in_channels = NULL, out_channels = 1L, packages = character(0), tags = NULL,
  classname = NULL, parent_env = parent.frame()) {
  assert_string(id)
  assert_function(initialize, null.ok = TRUE)
  assert_function(forward)
  assert_function(shapes_out)
  # `parent.frame()` has to be forced here: as the default of an argument that is only used further
  # down it would be evaluated in whichever frame happens to force it
  force(parent_env)

  module_generator = invoke(nn_module, classname = id, parent_env = parent_env,
    .args = discard(list(initialize = initialize, forward = forward), is.null))

  pipeop_torch_class(id = id, module_generator = module_generator, shapes_out = shapes_out,
    param_set = param_set, in_channels = in_channels, out_channels = out_channels,
    packages = packages, tags = tags, classname = classname)
}

#' @title Convert a Torch Module to a PipeOp
#'
#' @description
#' Converts an [`nn_module`][torch::nn_module] into a [`PipeOpTorch`], so that it can be used in a
#' [`Graph`][mlr3pipelines::Graph].
#' Everything the operator needs is taken from the module itself, which therefore has to be written
#' for this purpose:
#'
#' * `$initialize()` may take the arguments `shapes_in` (the input shapes, named after the input
#'   channels) and `task` (the [`Task`][mlr3::Task] or `NULL`) besides its own; they are supplied by
#'   the `PipeOp` and passed only if it declares them.
#'   Its remaining arguments become the hyperparameters of the `PipeOp`: untyped, and required if
#'   they have no default.
#' * `$shapes_out()` states the shapes of the tensors that `$forward()` produces, as a function of
#'   `shapes_in`, `param_vals` and `task`.
#'   It is called on the module *class* rather than on an instance -- there is no instance yet when
#'   the network is assembled -- so it cannot use `self`.
#' * the arguments of `$forward()` become the input channels; the `PipeOp`'s id is the module's
#'   class name.
#'
#' Use [`pipeop_torch()`] for a module that is not written this way, or when the operator needs more
#' than one output channel.
#'
#' @param x (`nn_module_generator`)\cr
#'   The module to convert.
#' @param clone (`logical(1)`)\cr
#'   Ignored, a new [`PipeOpTorch`] is created in either case.
#' @return A [`PipeOpTorch`].
#' @export
#' @examplesIf torch::torch_is_installed()
#' nn_scale = nn_module("nn_scale",
#'   initialize = function(shapes_in, init = 1) {
#'     self$weight = nn_parameter(torch_full(tail(shapes_in[[1L]], 1L), init))
#'   },
#'   forward = function(input) input * self$weight,
#'   shapes_out = function(shapes_in) shapes_in
#' )
#'
#' po_scale = as_pipeop(nn_scale)
#' po_scale$id
#' po_scale$param_set$ids()
#' po_scale$shapes_out(list(c(NA, 4)))
as_pipeop.nn_module_generator = function(x, clone = FALSE) { # nolint
  id = assert_string(x$classname, .var.name = "the module's class name")
  shapes_out = get_method(x, "shapes_out")
  if (is.null(shapes_out)) {
    stopf("The module '%s' has no `shapes_out()` method, so the shapes it produces are unknown. Add one, or use `pipeop_torch()` to state them separately.", id) # nolint
  }
  pipeop_torch_class(id = id, module_generator = x, shapes_out = shapes_out)$new()
}

pipeop_torch_class = function(id, module_generator, shapes_out, param_set = NULL,
  in_channels = NULL, out_channels = 1L, packages = character(0), tags = NULL,
  classname = NULL) {
  if (!is.null(param_set)) assert_param_set(param_set)
  assert_character(packages, any.missing = FALSE)
  assert_character(tags, null.ok = TRUE)
  assert_string(classname, null.ok = TRUE)

  # a module that only implements `$forward()` has no arguments to infer anything from
  init = get_method(module_generator, "initialize")
  init_args = (if (is.null(init)) NULL else names(formals(init))) %??% character(0)
  # the arguments the operator supplies itself, which are therefore not hyperparameters
  reserved = intersect(c("shapes_in", "task"), init_args)
  if (!is.null(param_set)) {
    assert_disjunct(param_set$ids(), reserved,
      .var.name = "parameter ids and the arguments supplied by the PipeOp")
    if ("..." %nin% init_args) {
      assert_subset(param_set$ids(), init_args, .var.name = "parameter ids")
    }
  }

  inname = channel_names(in_channels, prefix = "input", vararg = TRUE,
    forward = get_method(module_generator, "forward"))
  outname = channel_names(out_channels, prefix = "output")

  classname = classname %??% paste0("PipeOpTorch",
    paste0(capitalize(strsplit(sub("^nn_", "", id), split = "_")[[1L]]), collapse = ""))

  # the parameter set is built during construction so that each instance owns its own copy, and
  # `shapes_out()` is assigned rather than declared as a method, which would rebind its environment
  init_fun = crate(function(id = id, param_vals = list()) { # nolint
    info = private$.__construction_info
    private$.shapes_out_fn = info$shapes_out
    super$initialize(
      id = id,
      module_generator = info$module_generator,
      param_set = if (is.null(info$param_set)) {
        inferps(get_method(info$module_generator, "initialize"), ignore = info$reserved,
          required = TRUE)
      } else {
        info$param_set$clone(deep = TRUE)
      },
      param_vals = param_vals,
      inname = info$inname,
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
      shapes_out = shapes_out,
      reserved = reserved,
      inname = inname,
      outname = outname,
      packages = packages,
      tags = tags
    ),
    .reserved = reserved,
    .shapes_out_fn = NULL,
    .shapes_out = crate(function(shapes_in, param_vals, task) {
      shapes = invoke_declared(private$.shapes_out_fn, # nolint
        list(shapes_in = shapes_in, param_vals = param_vals, task = task))
      if (!test_list(shapes) || length(shapes) != nrow(self$output)) { # nolint
        stopf("The `shapes_out()` of PipeOp with id '%s' must return a list of %i shape(s), one per output channel.", # nolint
          self$id, nrow(self$output)) # nolint
      }
      shapes
    }, .parent = topenv()),
    # two operators generated from the same `id` are only the same if they do the same thing
    .additional_phash_input = crate(function() {
      fns = list(private$.shapes_out_fn, get_method(self$module_generator, "initialize"), # nolint
        get_method(self$module_generator, "forward")) # nolint
      c(list(self$input$name, self$output$name, self$param_set$ids()), # nolint
        map(fns, function(fn) if (!is.null(fn)) list(formals(fn), body(fn))))
    }, .parent = topenv())
  )

  if (length(reserved)) {
    # `shapes_in` and `task` are handed to the module's constructor only if it asks for them
    private$.shape_dependent_params = crate(function(shapes_in, param_vals, task) {
      c(param_vals, list(shapes_in = shapes_in, task = task)[private$.reserved]) # nolint
    }, .parent = topenv())
  }

  # the generated methods are mlr3torch's own, so they are resolved in its namespace; the user's
  # functions keep the environments they were written in
  R6Class(classname,
    inherit = PipeOpTorch,
    public = list(initialize = init_fun),
    private = private,
    parent_env = topenv(environment())
  )
}

# `shapes_in`, `param_vals` and `task` are passed by name and only to a function that declares
# them, so that an operator that needs nothing but the input shapes is written as
# `function(shapes_in)` rather than with three arguments of which it ignores two.
invoke_declared = function(fn, args) {
  fmls = names(formals(fn))
  if ("..." %nin% fmls) args = args[intersect(fmls, names(args))]
  do.call(fn, args)
}

# Channels are given either as names or as a count, `0` meaning a single vararg channel as elsewhere
# in mlr3pipelines. The input channels default to the arguments of the module's `$forward()`.
channel_names = function(x, prefix, vararg = FALSE, forward = NULL) {
  if (is.null(x)) {
    x = setdiff(names(formals(forward)), "...")
    return(if (length(x)) x else "...")
  }
  if (is.character(x)) {
    return(assert_names(x, type = "unique", .var.name = sprintf("%s channels", prefix)))
  }
  assert_int(x, lower = if (vararg) 0L else 1L, .var.name = sprintf("number of %s channels", prefix))
  if (x == 0L) return("...")
  paste0(prefix, if (x > 1L) seq_len(x))
}
