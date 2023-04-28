#' @title Autotest for PipeOpTorch
#' @description
#' This tests a [`PipeOpTorch`] that is embedded in a [`Graph`].
#' Note that the id of the [`PipeOp`] to test should be the default id.
#' It tests whether:
#' * the output tensor(s) have the correct shape(s)
#' * the generated module has the intended class
#' * the parameters are correctly implemented
#' * Some other basic checks
#' @details
#' TODO
#' @param graph ([`Graph`])\cr
#'   The graph that contains the [`PipeOpTorch`] to be tested.
#' @param id (`character(1)`)\cr
#'   The id of the [`PipeOpTorch`] to be tested.
#' @param task ([`Task`])\cr
#'   The task with which the [`PipeOpTorch`] will be tested.
#' @param module_class (`character(1)`)\cr
#'   The module class that is expected from the generated module of the [`PipeOpTorch`].
#'   Default is the [`PipeOp`]s `id`.
#' @param exclude_args (`character()`)\cr
#'   We check that the arguments of the module's forward function match the input channels except for those
#'   values specified as `exclude_args`.
#'
#' @return `TRUE` if the autotest passes, errs otherwise.
#' @export
autotest_pipeop_torch = function(graph, id, task, module_class = id, exclude_args = character(0)) {
  po_test = graph$pipeops[[id]]
  result = graph$train(task)
  md = result[[1]]

  modulegraph = md$graph
  po_module = modulegraph$pipeops[[id]]
  if (is.null(po_module$module)) {
    stop("No pipeop with id '%s' found in the graph, did you mistype the id?", id)
  }

  # (1) class of generated module is as expected
  expect_class(po_module, "PipeOpModule")
  expect_class(po_module$module, c(module_class, "nn_module"))

  # (2) argument names of forward function match names of input channels

  testthat::expect_equal(po_module$input$name, po_test$input$name)
  fargs = formalArgs(po_module$module$forward)
  if ("..." %nin% fargs) {
    # we omit the "..." case, because otherwise the merge_sum etc. would be trickier, i.e. we would
    # have to create a module with the specified number of input arguments on the fly.
    # Might change in the future
    inname = po_test$input$name
    missing = setdiff(fargs, inname)
    if (!length(missing)) {
      testthat::expect_equal(fargs, inname)
    } else {
      x1 = fargs[names(fargs) %nin% missing]
      x2 = inname[names(inname) %nin% missing]
      testthat::expect_true(all(x1 == x2))
    }
  }

  # Consistent default ids
  x1 = paste0(tolower(gsub("^PipeOpTorch", "", class(po_test)[[1L]])))
  x2 = gsub("_", "", gsub("^nn_", "", po_test$id))
  testthat::expect_true(x1 == x2)

  # (3) Forward call works and the shapes are correct
  # (i)input (p)ointer and (o)utput (p)ointer for the tested PipeOp

  # the code below is probably written reaaaally stupidly
  # The problem is that the outputs of the network have modified ids, i.e. `"output_<id>"`
  # When we want to pass a subset of the network outputs to $shapes_out() of po_test, we need to make sure that
  # the names are correct
  #
  # Assume we have a graph A -> B.
  # If we build a nn_graph from that, an additional A1 is introduced, so that we have A1 <- A -> B
  # the connection between A and A1 is a nop, but the names are different.
  # We need to match the names out the output A1 to the names of A and then get the input channels of B
  # (which is the pipeop we are testing)


  # input pointer (for po_test)
  ip = pmap(graph$edges[get("dst_id") == id, .(x = get("src_id"), y = get("src_channel"))], function(x, y) c(x, y))

  # output pointer (for po_test)
  op = pmap(graph$output[, c("op.id", "channel.name")], function(op.id, channel.name) c(op.id, channel.name)) # nolint
  pointers = append(ip, op)
  net = model_descriptor_to_module(md, output_pointers = pointers, list_output = TRUE)$to(device = "cpu")


  ds = task_dataset(task, md$ingress, device = "cpu")
  batch = ds$.getbatch(1)
  out = with_no_grad(invoke(net$forward, .args = batch$x))

  # TODO: Make this better understandable
  tmp = net$graph$output
  # outnamein are the output nodes that contain the tensors that are the input to the pipeop we are testing
  outnamein = tmp[get("op.id") != id, c("op.id", "channel.name")]
  # outnameout are the actual output nodes
  outnameout = tmp[get("op.id") == id, c("op.id", "channel.name")]

  # appending `"output_"` before is necessary, because this is how it is implemented when non-terminal nodes
  # are used in the output map, i.e. a new node `"output_<id>"` is added and a connection from `"<id>"` to `"output_<id>"`

  # layerin is the network's output which is as also the input to the pipeop we are testing
  layerin = out[paste0(outnamein$op.id, ".", outnamein$channel.name)]
  # layerout is the actual pipeop
  layerout = out[paste0(outnameout$op.id, ".", outnameout$channel.name)]

  # To verify the shapes, we need to map the names of the input channels of our po_test to the elements in layerin
  # so we can pass layerin (with correct names) to $shapes_out() of po_test
  # We cant pass layerin to po_test$shapes_out() because the names are wrong.

  # we rebuild out model_descriptor_to_module builds the ids for the non-terminal output channels
  # But buy doing so get get the information to which input channels of po_test they are mapped!
  tmp1 = net$graph$edges[get("dst_id") == id, ]
  channels = tmp1[get("dst_id") == id, "dst_channel"][[1L]]
  names(channels) = paste0("output", "_", tmp1$src_id, "_", tmp1$src_channel, ".", tmp1$src_channel)
  names(layerin) = channels[names(channels)]

  predicted = po_test$shapes_out(map(layerin, dim), task)
  observed = map(layerout, dim)
  test_shapes(predicted, observed)

  return(TRUE)
}

# Note that we don't simply compare the shapes for equality. The actually observed shape does not have NAs,
# so wherevery the predicted dimension is NA, the observed dimension can be anything.
test_shapes = function(predicted, observed) {
  # they are both lists
  if (length(predicted) != length(observed)){
    stopf("This should have been impossible!")
  }
  fail = FALSE
  for (i in seq_along(predicted)) {
    p = predicted[[i]]
    o = observed[[i]]
    if (length(p) != length(o)) {
      fail = TRUE
      break
    }
    ii = is.na(p)
    p[ii] = o[ii]
    fail = !isTRUE(all.equal(p, o))
  }

  if (fail) {
    stopf("Got outputs of shapes %s but $shapes_out() said %s.", collapse_char_list(observed),
      collapse_char_list(predicted))
  }
}


collapse_char_list = function(x) {
  x = lapply(x, function(x) paste0("(", paste(x, collapse = ", "), ")"))
  x = paste(x, collapse = ", ")
  return(x)
}

run_paramtest = function(learner, fun, exclude = character(), tag = NULL) {
  par_learner = learner$param_set$ids(tags = tag)
  if (checkmate::test_list(fun)) {
    # for xgboost we pass a character vector with info scraped from the web
    if (mlr3misc::some(fun, function(x) class(x) == "character")) {
      which = which(mlr3misc::map_lgl(fun, function(x) class(x) == "character"))
      par_package = fun[[which]]
      fun[[which]] = NULL
      other = unlist(lapply(fun, formalArgs))
      par_package = append(par_package, other)
    } else {
      par_package = unlist(lapply(fun, formalArgs))
    }
  } else {
    par_package = formalArgs(fun)
  }

  missing = setdiff(par_package, par_learner)
  missing = setdiff(missing, c(exclude, "..."))

  extra = setdiff(par_learner, par_package)
  extra = setdiff(extra, c(exclude, "..."))

  if (length(c(missing, extra)) == 0L) {
    return(TRUE)
  }

  merror = eerror = character(0)

  if (length(missing) > 0) {
    merror = sprintf("Missing parameters for learner '%s': %s",
      learner$id, paste0(missing, collapse = ", "))
  }

  if (length(extra) > 0) {
    eerror = sprintf("Extra parameters for learner '%s': %s",
      learner$id, paste0(extra, collapse = ", "))
  }

  error = paste(merror, eerror, sep = "\n")

  list(ok = FALSE, error = error, missing = missing, extra = extra)
}

#' @title Parameter Test
#' @description 
#' Tests that parameters are correctly implemented
#'
#' @param param_set ([`ParamSet`])\cr
#'   The parameter set to check.
#' @param fns (`list()` of `function`s)\cr
#'   The functions whose arguments the parameter set implements.
#' @param exclude (`character`)\cr
#'   The parameter ids and arguments of the functions that are exluded from checking.
#' @param exclude_defaults (`character()`)\cr 
#'   For which parameters the defaults should not be checked.
#'
#' @export
autotest_paramset = function(param_set, fns, exclude = character(0), exclude_defaults = character(0)) {
  if (!is.list(fns)) fns = list(fns)

  args = Reduce(c, lapply(fns, formalArgs))
  ids = param_set$ids()

  # The parameter ids

  missing = setdiff(args, c(ids, exclude, "..."))
  extra = setdiff(ids, c(args, exclude, "..."))

  info = list()

  if (length(missing) > 0) {
    info$merror = sprintf("Missing parameters: %s", paste0(missing, collapse = ", "))
  }

  if (length(extra) > 0) {
    info$eerror = sprintf("Extra parameters: %s", paste0(extra, collapse = ", "))
  }


  # The defaults
  # Note that we ony check the defaults that are not language expressions.

  ps_defaults = param_set$default
  ps_defaults[setdiff(param_set$ids(), names(ps_defaults))] = list(NULL)
  ps_defaults = ps_defaults[setdiff(names(ps_defaults), c(exclude, exclude_defaults))]

  fn_defaults = Reduce(c, lapply(fns, formals)) %??% list()
  fn_defaults[map_lgl(fn_defaults, function(x) as.character(x) == "")] = list(NULL)

  wrong_defaults = imap(ps_defaults, function(default, id) {
    if(!identical(fn_defaults[[id]], default)) id
  })
  wrong_defaults = unlist(Filter(function(x) !is.null(x), wrong_defaults))

  if (length(wrong_defaults)) {
    info$derror = sprintf("Wrong defaults: %s", paste0(wrong_defaults, collapse = ", "))
  }

  # unchecked_defaults = ps_defaults[setdiff(names(ps_defaults), names(fn_defaults))]

  info
}

# Helper function to convert a vector of probabilities to a matrix
#
# sometimes useful in tests, e.g., mlr3learners.partykit::LearnerClassifMob
# uses this in its tests to set up its custom prediction function for a mob
# version of a logit model
prob_vector_to_matrix = function(p, levs) {
  stopifnot(is.numeric(p))
  y = matrix(c(1 - p, p), ncol = 2L, nrow = length(p))
  colnames(y) = levs
  y
}

autotest_torch_callback = function(cb, init_args = list()) {
  # TODO: Check that man page exists
  cbgen = cb$generator
  expect_string(cb$id)
  expect_class(cbgen, "R6ClassGenerator")
  expect_class(cb$param_set, "ParamSet")
  init_fn = get_init(cb$callback)
  if (is.null(init_fn)) init_fn = function() NULL
  paramtest = autotest_paramset(cb$param_set, init_fn)


  # Check that the correct object comes out
  cbt = do.call(cbgen$new, args = init_args)
  expect_class(cbt, "CallbackTorch")

  implemented_stages = names(cbgen$public_methods)[grepl("^on_", names(cbgen$public_methods))]

  expect_subset(implemented_stages, mlr3torch_callback_stages)
}
