# @title Autotest for PipeOpTorch
# @description
# This tests a [`PipeOpTorch`] that is embedded in a [`Graph`][mlr3pipelines::Graph].
# Note that the id of the [`PipeOp`] to test should be the default id.
# It tests whether:
# * the output tensor(s) have the correct shape(s)
# * the generated module has the intended class
# * the parameters are correctly implemented
# * Some other basic checks
# @param graph ([`Graph`])\cr
#   The graph that contains the [`PipeOpTorch`] to be tested.
# @param id (`character(1)`)\cr
#   The id of the [`PipeOpTorch`] to be tested.
# @param task ([`Task`][mlr3::Task])\cr
#   The task with which the [`PipeOpTorch`] will be tested.
# @param module_class (`character(1)`)\cr
#   The module class that is expected from the generated module of the [`PipeOpTorch`].
#   Default is the [`PipeOp`]s `id`.
# @param exclude_args (`character()`)\cr
#   We check that the arguments of the module's forward function match the input channels except for those
#   values specified as `exclude_args`.
#
# @return `TRUE` if the autotest passes, errs otherwise.
# @export
expect_pipeop_torch = function(graph, id, task, module_class = id, exclude_args = character(0)) {
  require_namespaces(c("testthat"))
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
  ip = pmap(graph$edges[get("dst_id") == id, list(x = get("src_id"), y = get("src_channel"))], function(x, y) c(x, y))

  # output pointer (for po_test)
  op = pmap(graph$output[, c("op.id", "channel.name")], function(op.id, channel.name) c(op.id, channel.name)) # nolint
  pointers = append(ip, op)
  net = model_descriptor_to_module(md, output_pointers = pointers, list_output = TRUE)$to(device = "cpu")


  ds = task_dataset(task, md$ingress, device = "cpu")
  batch = ds$.getbatch(1)
  out = with_no_grad(invoke(net$forward, .args = batch$x))

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

  # check that shapes are compatible when passing batch dimension
  predicted = po_test$shapes_out(map(layerin, dim), task)
  observed = map(layerout, dim)
  expect_compatible_shapes(predicted, observed)

  # check that shapes are compatible without batch dimension as NA
  predicted_unknown_batch = po_test$shapes_out(map(layerin, function(d) {
    x = dim(d)
    x[1L] = NA
    x
  }), task)

  expect_error(assert_shapes(predicted_unknown_batch, null_ok = FALSE, unknown_batch = TRUE), regexp = NA)

  expect_compatible_shapes(predicted_unknown_batch, observed)


  # parameters must only ce active during training
  walk(po_test$param_set$tags, function(tags) {
    if (!(("train" %in% tags) && !("predict" %in% tags))) {
      stopf("Parameters of PipeOps inheriting from PipeOpTorch must only be active during training.")
    }
  })

  return(TRUE)
}

# Note that we don't simply compare the shapes for equality. The actually observed shape does not have NAs,
# so wherevery the predicted dimension is NA, the observed dimension can be anything.
expect_compatible_shapes = function(predicted, observed) {
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


# @title Parameter Test
# @description
# Tests that parameters are correctly implemented
# @param x ([`ParamSet`][paradox::ParamSet] or object with field `$param_set`)\cr
#   The parameter set to check.
# @param fns (`list()` of `function`s)\cr
#   The functions whose arguments the parameter set implements.
# @param exclude (`character`)\cr
#   The parameter ids and arguments of the functions that are exluded from checking.
# @param exclude_defaults (`character()`)\cr
#   For which parameters the defaults should not be checked.
expect_paramset = function(x, fns, exclude = character(0), exclude_defaults = character(0)) {
  if (test_r6(x, "ParamSet")) {
    param_set = x
  } else if (test_r6(x$param_set, "ParamSet")) {
    param_set = x$param_set
  } else {
    stopf("Argument 'x' is neither a ParamSet nor does it have a valid $param_set field.")
  }
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

  ps_defaults = param_set$default
  ps_defaults = ps_defaults[setdiff(names(ps_defaults), c(exclude, exclude_defaults))]

  fn_defaults = Reduce(c, lapply(fns, formals)) %??% list()

  args = setdiff(args, c(exclude, exclude_defaults, missing, extra))
  wrong_defaults = list()
  for (arg in args) {
    # This needs special treatment because of some weird behaviour in R
    # Consider formals(function(x) NULL)[[1]]
    # (therefore we can't assign fn_defaults[[arg]] to a variable)
    if (is.name(fn_defaults[[arg]]) && as.character(fn_defaults[[arg]]) == "") { # upstream has no default
      if (exists(arg, ps_defaults)) {
        # if ps implements a default it is a wrong default
        wrong_defaults = c(wrong_defaults, arg)
      }
    } else {
      # Here upstream has a default so the parameter must have a default and it must be identical
      # to the upstream default (here we need to be careful to not access list elements that don't
      # exist as we get NULL in return, therefore the exists(...) check)
      # Furthermore we need to be careful regaring expressions (such as -1, c("a", "b"))
      # because the defaults are then calls and not the evaluated expression.
      # we use the heuristic that either the value of the evaluated or unevaluated upstream default
      # must be identical. This can give false negatives but they should be rare

      if (!exists(arg, ps_defaults)) { # upstream has default but ps has none
        wrong_defaults = c(wrong_defaults, arg)
        next
      }

      upstream_default = fn_defaults[[arg]]
      upstream_default_eval = try(eval(upstream_default), silent = TRUE)
      implemented = ps_defaults[[arg]]

      ok = isTRUE(all.equal(upstream_default, implemented))
      if (!inherits(upstream_default_eval, "try-error")) {
        ok = ok || isTRUE(all.equal(upstream_default_eval, implemented))
      }
      if (!ok) {
        wrong_defaults = c(wrong_defaults, arg)
      }
    }
  }

  if (length(wrong_defaults)) {
    info$derror = sprintf("Wrong defaults: %s", paste0(wrong_defaults, collapse = ", "))
  }

  res = list(ok = identical(info, list()), info = info)
  return(res)
}

expect_paramtest = function(paramtest) {
  expect_true(paramtest$ok, info = paramtest$info)
}


#' @title Autotest for Torch Callback
#' @description
#' Performs various sanity checks on a callback descriptor.
#'
#' @param torch_callback ([`TorchCallback`])\cr
#'   The object to test.
#' @param check_man (`logical(1)`)\cr
#'   Whether to check that the manual page exists. Default is `TRUE`.
expect_torch_callback = function(torch_callback, check_man = TRUE) {
  # Checks on descriptor
  expect_class(torch_callback, "TorchCallback")
  expect_string(torch_callback$id)
  expect_string(torch_callback$label)
  expect_r6(torch_callback$param_set, "ParamSet")
  if (check_man) {
    expect_true(!is.null(torch_callback$man))
    expect_man_exists(torch_callback$man)
    expect_true(startsWith(torch_callback$man, "mlr3torch::mlr_callback_set"))
  }

  # Checks on generator
  cbgen = torch_callback$generator
  expect_class(cbgen, "R6ClassGenerator")
  expect_true(grepl("^CallbackSet", cbgen$classname))
  expect_true(cbgen$cloneable)
  init_fn = get_init(torch_callback$generator)
  if (is.null(init_fn)) init_fn = function() NULL
  paramtest = expect_paramset(torch_callback$param_set, init_fn)
  expect_paramtest(paramtest)
  implemented_stages = names(cbgen$public_methods)[grepl("^on_", names(cbgen$public_methods))]
  expect_subset(implemented_stages, mlr_reflections$torch$callback_stages)
  expect_true(length(implemented_stages) > 0)
  walk(implemented_stages, function(stage) expect_function(cbgen$public_methods[[stage]], nargs = 0))

  cb = torch_callback$generate()
  expect_deep_clone(cb, cb$clone(deep = TRUE))
}

#' @title Autotest for PipeOpTaskPreprocTorch
#' @description
#' Performs various sanity checks on a [`PipeOpTaskPreprocTorch`].
#' @param obj ([`PipeOpTaskPreprocTorch`])\cr
#'   The object to test.
#' @parm tnsr_in (`integer()`)\cr
expect_pipeop_torch_preprocess = function(obj, shapes_in, exclude = character(0), exclude_defaults = character(0),
  in_package = TRUE, seed = NULL, deterministic) {
  if (is.null(seed)) {
    seed = sample.int(100000, 1)
  }
  expect_pipeop(obj)
  expect_class(obj, "PipeOpTaskPreprocTorch")
  # a) Check that all parameters but stages have tags train and predict (this should hold in basically all cases)
  # parameters must only ce active during training
  walk(obj$param_set$tags, function(tags) {
    if (!(("train" %in% tags) && !("predict" %in% tags))) {
      stopf("Parameters of PipeOps inheriting from PipeOpTorch must only be active during training.")
    }
  })
  expect_paramset(obj$param_set, obj$fn, exclude = exclude, exclude_defaults = exclude_defaults)
  # b) Check that the shape prediction is compatible (already done in autotest for pipeop torch)

  # c) check that start with stages / trafo, depending on the initial value
  if (in_package) {
    class = get(class(obj)[[1L]], envir = getNamespace("mlr3torch"))
    instance = class$new()

    testthat::expect_true(grepl(instance$id, pattern = "^(trafo|augment)_"))
    if (startsWith(instance$id, "augment")) {
      expect_set_equal(instance$param_set$values$stages, "train")
    } else {
      expect_set_equal(instance$param_set$values$stages, "both")
    }
  }

  shapes_in = if (!test_list(shapes_in)) list(shapes_in) else shapes_in

  make_task = function(shape_in) {
    tnsr_in = torch_empty(shape_in)
    as_task_regr(data.table(
      y = rep(1, nrow(tnsr_in)),
      x = as_lazy_tensor(tnsr_in)
    ), target = "y")
  }

  walk(shapes_in, function(shape_in) {
    shape_out = obj$shapes_out(list(shape_in), stage = "train")[[1L]]
    shape_in_unknown_batch = shape_in
    shape_in_unknown_batch[1L] = NA
    shape_out_unknown_batch = obj$shapes_out(list(shape_in_unknown_batch), stage = "train")[[1L]]
    taskin = make_task(shape_in)

    # reproducible
    dout = obj$train(list(taskin))[[1L]]$data(cols = "x")[[1L]]
    tnsr_out1 = materialize(dout, rbind = FALSE)
    expect_list(tnsr_out1, types = "torch_tensor")

    if (!is.null(shape_out)) {
      tnsr_out1 = torch_cat(map(tnsr_out1, function(x) x$unsqueeze(1)), dim = 1L)
      testthat::expect_equal(tnsr_out1$shape, shape_out)
      testthat::expect_equal(tnsr_out1$shape[-1L], shape_out_unknown_batch[-1L])
      testthat::expect_true(is.na(shape_out_unknown_batch[1L]))
    }
  })

  if (deterministic) {
    # train
    taskin = make_task(shapes_in[[1L]])$clone(deep = TRUE)$filter(1:2)
    taskin1 = taskin$clone(deep = TRUE)$filter(1)
    taskin2 = taskin$clone(deep = TRUE)$filter(2)

    dtrain = with_torch_settings(seed = 1, expr = {
      obj$train(list(taskin))[[1L]]$data()
    })
    dtrain1 = with_torch_settings(seed = 1, expr = {
      obj$train(list(taskin1))[[1L]]$data()
    })
    dtrain2 = with_torch_settings(seed = 1, expr = {
      obj$train(list(taskin2))[[1L]]$data()
    })

    testthat::expect_equal(materialize(dtrain[1]), materialize(dtrain1))
    testthat::expect_equal(materialize(dtrain[2]), materialize(dtrain2))

    # predict
    taskin = make_task(shapes_in[[1L]])$clone(deep = TRUE)$filter(1:2)
    taskin1 = taskin$clone(deep = TRUE)$filter(1)
    taskin2 = taskin$clone(deep = TRUE)$filter(2)

    dtest = with_torch_settings(seed = 1, expr = {
      obj$predict(list(taskin))[[1L]]$data()
    })
    dtest1 = with_torch_settings(seed = 1, expr = {
      obj$predict(list(taskin1))[[1L]]$data()
    })
    dtest2 = with_torch_settings(seed = 1, expr = {
      obj$predict(list(taskin2))[[1L]]$data()
    })

    testthat::expect_equal(materialize(dtest[1]), materialize(dtest1))
    testthat::expect_equal(materialize(dtest[2]), materialize(dtest2))
  }

  # FIXME: test for test rows when available
}

expect_learner_torch = function(learner, task, check_man = TRUE, check_id = TRUE) {
  checkmate::expect_class(learner, "LearnerTorch")
  get("expect_learner", envir = .GlobalEnv)(learner)
  # state cloning is tested separately
  learner1 = learner
  learner1$state = NULL
  expect_deep_clone(learner1, learner1$clone(deep = TRUE))
  rr = resample(task, learner, rsmp("holdout"))
  expect_double(rr$aggregate())
  checkmate::expect_class(rr, "ResampleResult")
  if (check_id) testthat::expect_true(startsWith(learner$id, learner$task_type))
  checkmate::expect_subset(c("loss", "optimizer", "callbacks"), formalArgs(learner$initialize))
  checkmate::expect_subset(c("mlr3", "mlr3torch", "torch"), learner$packages)
  testthat::expect_true(all(map_lgl(learner$tags, function(tags) "predict" %in% tags || "train" %in% tags)))

  learner$param_set$set_values(device = "meta")
  ds = learner$dataset(task)
  batch = if (is.null(ds$.getbatch)) {
    ds$.getitem(1)
  } else {
    ds$.getbatch(1)
  }
  lapply(batch$x, function(tnsr) testthat::expect_true(tnsr$device == torch_device("meta")))
  testthat::expect_true(batch$y$device == torch_device("meta"))
  if (task$task_type == "regr") {
    testthat::expect_true(batch$y$dtype == torch_float())
  } else {
    testthat::expect_true(batch$y$dtype == torch_long())
  }
  testthat::expect_true(batch$.index$device == torch_device("meta"))
  testthat::expect_true(batch$.index$dtype == torch_long())
}
