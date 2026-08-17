#' @title Create a PipeOpTorch
#'
#' @description
#' Helper function to create a custom [`PipeOpTorch`] class for the most common cases.
#' A practical guide to this function is the article
#' [Writing your own PipeOpTorch](https://mlr3torch.mlr-org.com/articles/custom_pipeop_torch.html).
#' For more information and the more general case, see the *Inheriting* section of [`PipeOpTorch`].
#' The function works similarly to [`nn_module()`][torch::nn_module], except that
#' `$initialize()` can take two further arguments that the `PipeOp` supplies: `shapes_in` and `task`
#' and that it also needs to implement the method `$shapes_out()` which belongs to the `PipeOpTorch`.
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
#'   shape depends on nothing else is written as `function(shapes_in)`.
#'   Any dimension of `shapes_in` can be `NA`, i.e. unknown, so this function must not assume that a
#'   dimension it reads is known; see the *Inheriting* section of [`PipeOpTorch`].
#' @param param_set ([`ParamSet`][paradox::ParamSet] or `NULL`)\cr
#'   The parameter set.
#'   If left as `NULL` (default), it is inferred from the arguments of `initialize`:
#'   All arguments but `shapes_in` and `task` become an
#'   untyped parameter tagged `"train"`, and additionally `"required"` if it has no default.
#' @param in_channels (`character()` or `integer(1)` or `NULL`)\cr
#'   The input channels, either as names or as a count, where `0` means a single *vararg* channel.
#'   If `NULL` (default), the arguments of `forward` are used.
#' @param out_channels (`character()` or `integer(1)`)\cr
#'   The output channels, either as names or as a count, `1` by default.
#'   A module with more than one output channel must return a `list()`, in the order of the
#'   channels.
#' @param classname (`character(1)`)\cr
#'   The class name of the generated [`R6Class`][R6::R6Class].
#'   By default it is derived from `id`: a leading `"nn_"` is dropped, the remaining `_`-separated
#'   words are capitalized and pasted together, and `"PipeOpTorch"` is prepended, so the id
#'   `"nn_scale"` gives `"PipeOpTorchScale"`.
#' @param parent_env (`environment`)\cr
#'   The environment in which the module's methods are evaluated, the calling environment by
#'   default, as for [`nn_module()`][torch::nn_module].
#'   `initialize` and `forward` become methods of the module and are therefore evaluated in this
#'   environment rather than in the one they were written in; `shapes_out`, which is not a method of
#'   the module, keeps its own environment.
#'   The two only differ when the functions are written somewhere other than the caller of
#'   `pipeop_torch()`, e.g. in a function that wraps it.
#' @template param_packages
#' @param tags (`character()`)\cr
#'   Tags for the `PipeOp`. The tag `"torch"` is always added.
#'
#' @return An [`R6Class`][R6::R6Class] generator inheriting from [`PipeOpTorch`].
#'
#' @family Graph Network
#' @export
#' @examplesIf torch::torch_is_installed()
#' # A layer that scales its input by a learned factor
#' # Note that the number of features is read from the input shape
#' PipeOpTorchCustomScale = pipeop_torch("nn_custom_scale",
#'   initialize = function(shapes_in, init = 1) {
#'     self$weight = nn_parameter(torch_full(tail(shapes_in[[1L]], 1L), init))
#'   },
#'   forward = function(input) input * self$weight,
#'   shapes_out = function(shapes_in) shapes_in # scaling leaves the shape as it is
#' )
#'
#' po_custom_scale = PipeOpTorchCustomScale$new()
#' # `init` is a hyperparameter, the number of features is not
#' po_custom_scale$param_set$ids()
#' po_custom_scale$shapes_out(list(c(NA, 4)))
#' # the operator can now be used like any other, and the module is built with 4 features
#' md = po("torch_ingress_num") %>>% po_custom_scale %>>% po("nn_head")
#' network = model_descriptor_to_module(md$train(tsk("iris"))[[1L]])
#' network
#'
 
#' # To use it via `nn("custom_scale")` or `po("nn_custom_scale")` we could run the line below:
#' # mlr_pipeops$add("nn_custom_scale", PipeOpTorchCustomScale)
pipeop_torch = function(id, initialize = NULL, forward, shapes_out, param_set = NULL,
  in_channels = NULL, out_channels = 1L, packages = character(0), tags = NULL,
  classname = NULL, parent_env = parent.frame()) {
  assert_string(id, min.chars = 1L)
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

pipeop_torch_class = function(id, module_generator, shapes_out, param_set = NULL,
  in_channels = NULL, out_channels = 1L, packages = character(0), tags = NULL,
  classname = NULL) {
  assert_string(id, min.chars = 1L)
  if ("..." %nin% names(formals(shapes_out))) {
    assert_subset(names(formals(shapes_out)), c("shapes_in", "param_vals", "task"),
      .var.name = "arguments of `shapes_out()`")
  }
  if (!is.null(param_set)) assert_param_set(param_set)
  assert_character(packages, any.missing = FALSE)
  assert_character(tags, null.ok = TRUE)
  assert_string(classname, null.ok = TRUE)

  init = get_method(module_generator, "initialize")
  init_args = (if (is.null(init)) NULL else names(formals(init))) %??% character(0)
  reserved = intersect(c("shapes_in", "task"), init_args)
  if (!is.null(param_set)) {
    assert_disjunct(param_set$ids(), reserved,
      .var.name = "parameter ids and the arguments supplied by the PipeOp")
    if ("..." %nin% init_args) {
      assert_subset(param_set$ids(), init_args, .var.name = "parameter ids")
    }
    missing_ids = setdiff(setdiff(required_args(init), reserved), param_set$ids())
    if (length(missing_ids)) {
      stopf("The `param_set` has no parameter(s) %s, which the module's `initialize()` requires.",
        paste0("'", missing_ids, "'", collapse = ", "))
    }
  }

  forward = get_method(module_generator, "forward")
  inname = channel_names(in_channels, prefix = "input", vararg = TRUE, forward = forward)
  outname = channel_names(out_channels, prefix = "output")
  assert_channels_match_forward(inname, forward)

  classname = classname %??% derive_classname(id)

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
      invoke_declared(private$.shapes_out_fn, # nolint
        list(shapes_in = shapes_in, param_vals = param_vals, task = task))
    }, .parent = topenv()),
    # two operators generated from the same `id` are only the same if they do the same thing
    .additional_phash_input = crate(function() {
      list(self$input$name, self$output$name, self$param_set$ids(), # nolint
        private$.shapes_out_fn, self$module_generator) # nolint
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
    assert_names(x, type = "unique", .var.name = sprintf("%s channels", prefix))
    # the counts below only allow a vararg where one is meaningful, so the names must do the same
    if ("..." %in% x) {
      if (!vararg) {
        stopf("'...' is not a valid %s channel name, only input channels can be vararg.", prefix)
      }
      if (length(x) > 1L) {
        stopf("A vararg channel ('...') must be the only %s channel.", prefix)
      }
    }
    return(x)
  }
  assert_int(x, lower = if (vararg) 0L else 1L, .var.name = sprintf("number of %s channels", prefix))
  if (x == 0L) return("...")
  paste0(prefix, if (x > 1L) seq_len(x))
}

# The module is called with one argument per input channel, so a mismatch would otherwise only
# surface when the network is run, as an `unused argument` error that names neither the module nor
# the channel. Arguments with a default may be left to it, hence the range.
assert_channels_match_forward = function(inname, forward) {
  if (is.null(forward) || "..." %in% inname) return(invisible(inname))
  fmls = formals(forward)
  if ("..." %in% names(fmls)) return(invisible(inname))
  n_required = length(required_args(forward))
  if (length(inname) > length(fmls) || length(inname) < n_required) {
    stopf("The module's `forward()` takes %i argument(s) (%s), but %i input channel(s) (%s) were given.",
      length(fmls), paste0(names(fmls), collapse = ", "), length(inname), paste0(inname, collapse = ", "))
  }
  invisible(inname)
}

# the names of the arguments of `fn` that have no default
required_args = function(fn) {
  if (is.null(fn)) return(character(0))
  fmls = formals(fn)
  setdiff(names(fmls)[map_lgl(fmls, function(arg) identical(arg, quote(expr = )))], "...")
}

# `"nn_multihead_attention"` becomes `"PipeOpTorchMultiheadAttention"`
derive_classname = function(id) {
  words = Filter(nzchar, strsplit(sub("^nn_", "", id), split = "_")[[1L]])
  if (!length(words)) {
    stopf("No class name can be derived from the id '%s', pass `classname` explicitly.", id)
  }
  paste0("PipeOpTorch", paste0(capitalize(words), collapse = ""))
}
