# Randomized verification of the shape inference: for every operator a number of (parameter,
# shape) combinations is drawn, the shape inference is run, the module is built from the -- possibly
# partially unknown -- shape and run on a tensor, and the two are compared. See
# `helper_shape_inference.R`; the number of draws per operator is controlled by
# `MLR3TORCH_SHAPE_BUDGET` and is deliberately small by default so that CI stays fast.

test_that("shape inference of every PipeOpTorch agrees with its module", {
  specs = shape_inference_specs()
  budget = shape_inference_budget()

  # every PipeOpTorch that is not listed is shape-preserving and needs no parameters
  # the categorical tokenizer needs an integer tensor and a task, so it is checked separately
  skip = c("nn_tokenizer_categ", "nn_block", "nn_fn")
  shape_preserving = Filter(function(key) {
    obj = suppressWarnings(try(po(key), silent = TRUE))
    !inherits(obj, "try-error") && inherits(obj, "PipeOpTorch") && key %nin% c(names(specs), skip) &&
      !length(obj$param_set$ids(tags = "required"))
  }, mlr_pipeops$keys())
  specs = c(specs, set_names(lapply(shape_preserving, function(key) {
    list(rank = 3L, params = function() list())
  }), shape_preserving))

  expect_true(length(specs) >= 50L)
  for (id in names(specs)) {
    expect_shape_inference_sampled(id, specs[[id]], budget = budget)
  }
})

test_that("shape inference of every preprocessing PipeOp agrees with its function", {
  specs = preproc_inference_specs()
  budget = shape_inference_budget()
  for (id in names(specs)) {
    expect_shape_inference_sampled_preproc(id, specs[[id]], budget = budget)
  }

  # the remaining preprocessing operators declare an unknown output shape, which is always safe
  unknown = Filter(function(key) {
    obj = suppressWarnings(try(po(key), silent = TRUE))
    !inherits(obj, "try-error") && inherits(obj, "PipeOpTaskPreprocTorch") && key %nin% names(specs)
  }, mlr_pipeops$keys())
  for (id in unknown) {
    obj = po(id)
    # operators with unset required parameters cannot be asked for their shapes
    shape = tryCatch(obj$shapes_out(list(c(NA, 3L, 8L, 8L)), stage = "train")[[1L]],
      error = function(e) "skip")
    if (identical(shape, "skip")) next
    expect_true(is.null(shape), label = sprintf("%s declares an unknown output shape", id))
  }
})
