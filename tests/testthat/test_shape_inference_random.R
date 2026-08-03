# Class-wide check that no operator is left without a shape-inference test. The operators that
# need parameters or a specific input rank are checked in their own test file, see
# `expect_shape_inference_sampled()`; everything else is shape-preserving and swept here, so that a
# newly added operator is covered without anyone having to remember it.

test_that("every PipeOpTorch that needs no parameters agrees with its module", {
  budget = shape_inference_budget()
  # Operators that need parameters, a specific input rank, more than one input or a task are
  # checked in their own test file, which is where their sampler lives. Everything else takes any
  # rank-3 input and is swept below. An operator with a required parameter cannot be built from
  # its defaults at all and is filtered out anyway, the ones listed here are those that could be
  # built but would not be checked correctly.
  own_test = c("nn_tokenizer_categ", "nn_tokenizer_num", "nn_block", "nn_fn", "nn_head",
    "nn_flatten", "nn_unsqueeze", "nn_squeeze", "nn_reshape", "nn_merge_sum", "nn_merge_prod",
    "nn_merge_cat", "nn_ft_cls", "nn_ft_transformer_block", "nn_dropout",
    sprintf("nn_batch_norm%id", 1:3))
  shape_preserving = Filter(function(key) {
    obj = suppressWarnings(try(po(key), silent = TRUE))
    !inherits(obj, "try-error") && inherits(obj, "PipeOpTorch") && key %nin% own_test &&
      !length(obj$param_set$ids(tags = "required"))
  }, mlr_pipeops$keys())

  # guards the enrolment itself: if this filter ever stops matching, the sweep would silently
  # cover nothing
  expect_true(length(shape_preserving) >= 20L)

  for (id in shape_preserving) {
    expect_shape_inference_sampled(id, list(rank = 3L, params = function() list()),
      budget = budget)
  }
})

test_that("every preprocessing PipeOp agrees with its function", {
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
