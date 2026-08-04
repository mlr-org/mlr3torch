# Test helpers for shape inference

run_on_shape = function(fn, shape, n_in = 1L, device = c("meta", "cpu")) {
  make = function(device) {
    tensor = mlr3misc::invoke(torch_empty, .args = as.list(as.integer(shape)),
      device = torch_device(device))
    rep(list(tensor), n_in)
  }
  out = if (identical(device[1L], "cpu")) {
    with_no_grad(mlr3misc::invoke(fn, .args = make("cpu")))
  } else {
    tryCatch(
      with_no_grad(mlr3misc::invoke(fn, .args = make("meta"))),
      error = function(e) with_no_grad(mlr3misc::invoke(fn, .args = make("cpu")))
    )
  }
  # an operator with more than one output channel returns a `list()` of tensors, so the shapes are
  # always reported as a list to keep the single- and multi-channel cases the same downstream
  if (inherits(out, "torch_tensor")) list(dim(out)) else map(out, dim)
}

# Ground truth for a `PipeOpTorch`. The module is obtained the way a network gets it: the `PipeOp`
# is trained on `ModelDescriptor`s announcing `shape_in` -- the possibly partially unknown shape
# that the inference saw -- and the module it put into the graph is then run on a tensor of the
# concrete `shape`. This goes through the public `$train()` rather than `private$.make_module()`,
# so that the test exercises the same path a real graph does.
true_shape_torch = function(obj, shape_in, shape, n_in = 1L, task = NULL) {
  shape_in = as.integer(shape_in)
  # the module is built from exactly the shape the inference saw, batch dimension included, so that
  # an operator reading the batch size is held to what it claimed rather than to a blanked shape
  # the task is only read by operators that size themselves from it, such as `nn_head`
  task = task %??% tsk("iris")
  mds = map(seq_len(n_in), function(i) {
    nop = paste0("nop", i)
    ModelDescriptor(
      graph = as_graph(po("nop", id = nop)),
      ingress = set_names(list(TorchIngressToken("placeholder", function(data, ...) NULL, shape_in)),
        paste0(nop, ".input")),
      task = task,
      pointer = c(nop, "output"),
      pointer_shape = shape_in
    )
  })
  # a vararg channel takes the inputs unnamed, a fixed one wants them named after the channels
  if ("..." %nin% obj$input$name) mds = set_names(mds, obj$input$name)
  mdout = obj$train(mds)[[1L]]
  run_on_shape(mdout$graph$pipeops[[obj$id]]$module, shape, n_in)
}

# A minimal task whose single `lazy_tensor` column holds tensors of `shape`. A preprocessing
# `PipeOp` has to be trained on one before it can be asked for its predict shapes, because those
# are computed from the parameter values it recorded rather than from the ones currently set.
task_for_shape = function(shape) {
  shape = as.integer(shape)
  lt = as_lazy_tensor(torch_randn(shape))
  as_task_regr(data.table::data.table(y = as.double(seq_len(shape[1L])), x = lt), target = "y")
}

# Whether a preprocessing operator runs at all at `stage`; the augmentations default to training
# only, which makes them the identity at predict time.
preproc_runs_at = function(obj, stage) {
  stages = obj$param_set$values$stages
  if (identical(stages, "both")) stages = c("train", "predict")
  stage %in% stages
}

# ground truth for a `PipeOpTaskPreprocTorch`: apply its function, respecting `rowwise`
true_shape_preproc = function(obj, shape) {
  pv = obj$param_set$get_values(tags = "train")
  pv$stages = NULL
  pv$affect_columns = NULL
  fn = obj$fn
  args = pv[intersect(names(pv), names(formals(fn)))]
  f = function(x) mlr3misc::invoke(fn, x, .args = args)
  # a preprocessing operator has exactly one output channel, but the shape is still wrapped in a
  # list so that both kinds of operator report their shapes the same way
  if (obj$rowwise) {
    list(c(as.integer(shape[1L]), run_on_shape(f, shape[-1L], device = "cpu")[[1L]]))
  } else {
    run_on_shape(f, shape, device = "cpu")
  }
}

expect_shape_case = function(shape, inferred_shape, true_shape, label) {
  shape = as.integer(shape)
  # Both sides report one shape per output channel, so every channel is compared: an operator such
  # as `nn_multihead_attention` with `need_weights = TRUE` computes its second shape separately and
  # would otherwise go unchecked. The inferred shapes are named after the output channels and the
  # ground truth is not, so the comparison is on the shapes alone, in channel order.
  shapes_of = function(x) unname(map(x, as.integer))
  expect_equal(shapes_of(inferred_shape(shape)), shapes_of(true_shape(shape, shape)),
    label = sprintf("%s: known shape %s", label, shape_to_str(shape)))

  na_idx = c(as.list(seq_along(shape)), lapply(seq_along(shape)[-1L], function(i) c(1L, i)))
  for (idx in na_idx) {
    partial = shape
    partial[idx] = NA_integer_
    inferred = tryCatch(inferred_shape(partial), error = function(e) e)
    if (inherits(inferred, "condition")) {
      next
    }
    truth = true_shape(partial, shape)
    expect_equal(length(inferred), length(truth),
      label = sprintf("%s: number of output channels for %s", label, shape_to_str(partial)))
    for (k in seq_along(truth)) {
      what = sprintf("%s%s", shape_to_str(partial),
        if (length(truth) > 1L) sprintf(", channel %i", k) else "")
      expect_equal(length(inferred[[k]]), length(truth[[k]]),
        label = sprintf("%s: number of dimensions for %s", label, what))
      known = !is.na(inferred[[k]])
      expect_equal(as.integer(inferred[[k]])[known], as.integer(truth[[k]])[known],
        label = sprintf("%s: known dimensions for %s", label, what))
    }
  }
}

# --- sampling ------------------------------------------------------------------------------
#
# How many shapes each generator draws per PipeOp. Every draw additionally sweeps all
# single-`NA` patterns and the batch-plus-one patterns, so a budget of 3 already means dozens of
# comparisons per operator. Raise it while working on the shape inference:
#     MLR3TORCH_SHAPE_BUDGET=50 Rscript -e 'testthat::test_local()'
shape_inference_budget = function(default = 3L) {
  budget = Sys.getenv("MLR3TORCH_SHAPE_BUDGET")
  if (!nzchar(budget)) {
    return(default)
  }
  assert_int(as.integer(budget), lower = 1L)
}

# draws a dimension size for the sampled shapes
size = function(n = 1L, from = 4L, to = 12L) sample(seq(from, to), n, replace = TRUE)

# Makes the seed of a case, so that a failure can be reproduced by re-running the file: it is
# derived from the operator id and the index of the case. The concrete shapes use negative indices,
# which keeps their seeds apart from the drawn ones and independent of the budget.
inference_seed = function(id, i) sum(utf8ToInt(id)) + i

# Builds a shape generator: a function of no arguments that draws one input shape. `rank` is the
# number of dimensions, of which the first is a small batch dimension. The remaining ones are
# doubled by default, because the gated units halve a dimension and reshaping needs a divisible
# number of elements; `even = FALSE` turns that off.
gen_shape = function(rank = 3L, even = TRUE) {
  force(rank)
  force(even)
  function() {
    shape = c(sample(1:3, 1L), size(rank - 1L))
    if (even) shape[-1L] = shape[-1L] * 2L
    shape
  }
}



# The single entry point: it compares the inferred output shape of one operator against the shape
# the operator actually produces.
# The shapes can either be specified concretely or via a generator function that respects
# a global budget.
# Whether the operator is a `PipeOpTorch` or a `PipeOpTaskPreprocTorch` is decided from the
# constructed object: the two differ in how the output shape is asked for and in what the ground
# truth is, but not in what has to hold.
#
# @param id (`character(1)`) The key of the operator.
# @param params (named `list()` | `function()`) The parameter values, either fixed or a function
#   drawing them; the latter is redrawn for every case.
# @param shapes (`list()` of shapes | one shape) Concrete input shapes, always checked.
# @param generators (`function()` | `list()` of them) Functions drawing an input shape, see
#   `gen_shape()`. Each is used `budget` times.
# @param n_in (`integer(1)`) The number of input channels the shape is repeated over.
# @param task ([`Task`]) The task to pass to `$shapes_out()` and to the module construction.
# @param budget (`integer(1)`) How many shapes to draw per generator.
expect_shape_inference = function(id, params = list(), shapes = list(), generators = list(),
  n_in = 1L, task = NULL, budget = shape_inference_budget()) {
  if (!is.list(shapes)) shapes = list(shapes)
  if (is.function(generators)) generators = list(generators)

  check = function(param_vals, shape) {
    obj = do.call(po, c(list(id), param_vals))
    # parameter values may be functions (`nn_module_generator`s), so only their names are shown
    label = sprintf("%s(%s)", id, paste(names(param_vals), collapse = ", "))
    if (!inherits(obj, "PipeOpTaskPreprocTorch")) {
      return(expect_shape_case(shape,
        inferred_shape = function(s) obj$shapes_out(rep(list(s), n_in), task = task),
        true_shape = function(shape_in, s) true_shape_torch(obj, shape_in, s, n_in = n_in, task = task),
        label = label
      ))
    }
    # a preprocessing operator does not build a module, its function is applied directly and does
    # not depend on the input shape
    expect_shape_case(shape,
      inferred_shape = function(s) obj$shapes_out(list(s), stage = "train"),
      true_shape = function(shape_in, s) true_shape_preproc(obj, s),
      label = paste(label, "at train")
    )
    # The predict shapes are a separate question: they are only available once the operator has
    # been trained, and an operator that does not run at predict time -- the default for the
    # augmentations -- must leave the shape alone rather than apply its function.
    invisible(capture.output(obj$train(list(task_for_shape(shape)))))
    expect_shape_case(shape,
      inferred_shape = function(s) obj$shapes_out(list(s), stage = "predict"),
      true_shape = if (preproc_runs_at(obj, "predict")) {
        function(shape_in, s) true_shape_preproc(obj, s)
      } else {
        function(shape_in, s) list(as.integer(s))
      },
      label = paste(label, "at predict")
    )
  }
  draw_params = function() if (is.function(params)) params() else params

  for (j in seq_along(shapes)) {
    withr::local_seed(inference_seed(id, -j))
    check(draw_params(), shapes[[j]])
  }
  for (k in seq_along(generators)) {
    gen = generators[[k]]
    for (b in seq_len(budget)) {
      # the index runs across the generators, so that two of them do not draw the same shapes
      withr::local_seed(inference_seed(id, (k - 1L) * budget + b))
      # the shape is drawn before the parameters, so that adding a parameter to the draw does not
      # change which shapes are checked
      shape = gen()
      check(draw_params(), shape)
    }
  }
}
