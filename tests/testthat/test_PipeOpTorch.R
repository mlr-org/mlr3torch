test_that("Basic checks", {
  task = tsk("german_credit")

  # basic checks that output is checked correctly
  obj = PipeOpTorchDebug$new(id = "debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2))
  expect_pipeop(obj)
  expect_class(obj, "PipeOpTorch")

  expect_equal(unique(obj$input$train), "ModelDescriptor")
  expect_equal(unique(obj$output$train), "ModelDescriptor")

  expect_equal(unique(obj$input$predict), "Task")
  expect_equal(unique(obj$output$predict), "Task")
  expect_class(obj$module_generator, "nn_module_generator")
  expect_equal(obj$tags, "torch")
  expect_set_equal(obj$packages, c("mlr3torch", "torch", "mlr3pipelines"))
})

test_that("cloning works", {
  # can't use PipeOpTorchDebug because it then fails for some reason because of a  missing help page
  obj = PipeOpTorchReLU$new()
  obj1 = obj$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(obj, obj1)
})

test_that("single input and output", {
  task = tsk("iris")

  # train
  md = (po("torch_ingress_num") %>>%
    po("torch_optimizer") %>>%
    po("torch_loss", "cross_entropy") %>>%
    po("torch_callbacks", "checkpoint"))$train(task)[[1L]]

  obj = po("nn_linear", out_features = 10)

  mdout = obj$train(list(md))[[1L]]
  expect_identical(address(md$graph), address(mdout$graph))
  expect_true(!identical(md$pointer, mdout$pointer))
  expect_true(!identical(md$pointer_shape, mdout$pointer_shape))
  expect_equal(address(md$loss), address(mdout$loss))
  expect_equal(address(md$optimizer), address(mdout$optimizer))
  expect_equal(address(md$callbacks[[1L]]), address(mdout$callbacks[[1L]]))
  expect_equal(mdout$pointer, c("nn_linear", "output"))
  expect_equal(mdout$pointer_shape, c(NA, 10))
  expect_true(obj$is_trained)
  expect_true("nn_linear" %in% names(mdout$graph$pipeops))
  expect_class(mdout$graph$pipeops$nn_linear, "PipeOpModule")
  expect_class(mdout$graph$pipeops$nn_linear$module, "nn_linear")
  expect_equal(
    data.table(
      src_id = "torch_ingress_num",
      src_channel = "output",
      dst_id = "nn_linear",
      dst_channel = "input"
    ),
    mdout$graph$edges
  )

  # predict
  taskout = obj$predict(list(task))
  expect_identical(address(taskout[[1L]]), address(taskout[[1L]]))
})

test_that("train handles multiple input channels correctly", {
  task = tsk("iris")

  # first we start with vararg
  obj = po("nn_merge_sum")

  graph = as_graph(list(
    po("select_1", selector = selector_grep("Sepal")) %>>% po("torch_ingress_num_1"),
    po("select_2", selector = selector_grep("Petal")) %>>% po("torch_ingress_num_2"))
  )

  mds = graph$train(task)
  mdsout = obj$train(mds)
  expect_true(obj$is_trained)
  expect_equal(address(mdsout[[1L]]$graph), address(mdsout[[1L]]$graph))
  expect_equal(mdsout[[1L]]$pointer, c("nn_merge_sum", "output"))
  expect_equal(mdsout[[1L]]$pointer_shape, c(NA, 2))

  expect_equal(
    data.table(
      src_id = c("torch_ingress_num_1", "torch_ingress_num_2"),
      src_channel = c("output", "output"),
      dst_id = "nn_merge_sum",
      dst_channel = c("...", "...")
    ),
    mdsout[[1L]]$graph$edges
  )


  # two inputs two outputs


  obj = PipeOpTorchDebug$new(id = "nn_debug", inname = paste0("input", 1:2), outname = paste0("output", 1:2))
  obj$param_set$set_values(d_out1 = 2, d_out2 = 3, bias = TRUE)

  mdin1 = (po("select", selector = selector_grep("Petal")) %>>% po("torch_ingress_num_1"))$train(task)[[1L]]
  mdin2 = (po("select", selector = selector_grep("Sepal")) %>>% po("torch_ingress_num_2"))$train(task)[[1L]]

  mdouts = obj$train(list(input1 = mdin1, input2 = mdin2))
  mdout1 = mdouts[["output1"]]
  mdout2 = mdouts[["output2"]]

  expect_equal(address(mdout1$graph), address(mdout2$graph))
  expect_equal(mdout1$pointer, c("nn_debug", "output1"))
  expect_equal(mdout2$pointer, c("nn_debug", "output2"))
  expect_equal(mdout1$pointer_shape, c(NA, 2))
  expect_equal(mdout2$pointer_shape, c(NA, 3))
})

test_that("shapes_out", {
  obj = po("nn_linear", out_features = 3)

  # single input
  expect_equal(obj$shapes_out(c(NA, 1)), list(output = c(NA, 3)))
  expect_equal(obj$shapes_out(list(c(NA, 1))), list(output = c(NA, 3)))
  expect_equal(obj$shapes_out(list(input = c(NA, 1))), list(output = c(NA, 3)))
  expect_equal(obj$shapes_out(list(x = c(NA, 1))), list(output = c(NA, 3)))

  # multiple inputs
  obj1 = PipeOpTorchDebug$new()
  obj1$param_set$set_values(d_out1 = 2, d_out2 = 3)

  expect_equal(obj1$shapes_out(list(c(NA, 99), c(NA, 3))), list(output1 = c(NA, 2), output2 = c(NA, 3)))
  expect_error(obj1$shapes_out(list(c(NA, 99))), regexp = "but 1 input shape\\(s\\) were given")
})

test_that("Multiple NAs are allowed in the shape", {
  graph = as_graph(po("torch_ingress_num"))

  task = tsk("iris")
  md = graph$train(task)[[1L]]

  md$pointer_shape = c(4, NA)
  md = po("nn_relu")$train(list(md))[[1L]]
  expect_equal(md$pointer_shape, c(4, NA))

  md$pointer_shape = c(NA, NA, 4)
  expect_equal(md$pointer_shape, c(NA, NA, 4))
})

test_that("unknown dimensions other than the batch dimension are allowed", {
  obj = nn("linear", out_features = 10)
  expect_equal(obj$shapes_out(list(c(NA, NA, 1))), list(output = c(NA, NA, 10)))
})

test_that("NA in second dimension", {
  ds = dataset(
    initialize = function() {
      self$xs = lapply(1:10, function(i) torch_randn(sample(1:10, 1), 3))
    },
    .getitem = function(i) {
      list(x = self$xs[[i]])
    },
    .length = function() {
      length(self$xs)
    }
  )()

  task = as_task_regr(data.table(
    x = as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, NA, 3))),
    y = rnorm(10)
  ), target = "y", id = "test")

  graph = po("torch_ingress_ltnsr") %>>% po("nn_linear", out_features = 10)

  md = graph$train(task)[[1L]]

  expect_equal(md$pointer_shape, c(NA, NA, 10))

  net = model_descriptor_to_module(md)
  expect_equal(net(torch_randn(1, 2, 3))$shape, c(1, 2, 10))
  expect_equal(net(torch_randn(2, 1, 3))$shape, c(2, 1, 10))
})

test_that("$outputs restricts the output channels to a subset of the module's outputs", {
  po_pool = nn("max_pool1d", kernel_size = 2, outputs = c("output", "indices"))
  expect_equal(po_pool$outputs, c("output", "indices"))

  po_pool$outputs = "indices"
  expect_equal(po_pool$output$name, "indices")
  shapes = po_pool$shapes_out(list(c(NA, 3, 10)))
  expect_equal(names(shapes), "indices")
  expect_equal(shapes$indices, c(NA, 3L, 5L))

  # the channels keep the order of the module's outputs
  po_pool$outputs = c("indices", "output")
  expect_equal(po_pool$outputs, c("output", "indices"))

  expect_error(po_pool$outputs <- "foo", "subset")
  expect_error(po_pool$outputs <- character(0), "outputs")
})

test_that("$outputs can be set via nn() and po()", {
  expect_equal(nn("max_pool1d", kernel_size = 2, outputs = "indices")$outputs, "indices")
  expect_equal(po("nn_max_pool1d", kernel_size = 2, outputs = "indices")$outputs, "indices")
})

test_that("$outputs only enters the hash when outputs are left out", {
  all_outputs = nn("max_pool1d", kernel_size = 2, outputs = c("output", "indices"))
  some_outputs = nn("max_pool1d", kernel_size = 2, outputs = "output")
  expect_false(all_outputs$hash == some_outputs$hash)
  expect_false(all_outputs$phash == some_outputs$phash)
  some_outputs$outputs = c("output", "indices")
  expect_equal(all_outputs$hash, some_outputs$hash)
  expect_equal(all_outputs$phash, some_outputs$phash)
})

test_that("$outputs survives cloning", {
  po_pool = nn("max_pool1d", kernel_size = 2, outputs = "indices")
  expect_equal(po_pool$clone(deep = TRUE)$outputs, "indices")
})

test_that("a PipeOp with restricted outputs can be chained and trained", {
  task = tsk("iris")
  # without restricting the outputs, the indices of the pooling would need to go somewhere
  graph = po("torch_ingress_num") %>>% nn("unsqueeze", dim = 2) %>>%
    nn("max_pool1d", kernel_size = 2, outputs = "output") %>>%
    nn("flatten") %>>% nn("head") %>>% po("torch_loss", "cross_entropy") %>>%
    po("torch_optimizer", "adam") %>>% po("torch_model_classif", epochs = 1L, batch_size = 50L)
  learner = as_learner(graph)
  learner$train(task)
  expect_class(learner$predict(task), "PredictionClassif")
})

test_that("an output that is left out is not computed when the module allows for it", {
  task = tsk("iris")
  build = function(...) {
    graph = po("torch_ingress_num") %>>% nn("unsqueeze", dim = 2) %>>%
      nn("multihead_attention", num_heads = 1, ...)
    graph$train(task)
  }
  md = build(outputs = "output")
  expect_length(md, 1L)
  expect_false(md[[1L]]$graph$pipeops$multihead_attention$module$need_weights)
  net = model_descriptor_to_module(md[[1L]])
  expect_equal(net(torch_randn(2, 4))$shape, c(2, 1, 4))

  md = build(outputs = "weights")
  expect_true(md[[1L]]$graph$pipeops$multihead_attention$module$need_weights)
  net = model_descriptor_to_module(md[[1L]])
  expect_equal(net(torch_randn(2, 4))$shape, c(2, 1, 1))

  # torch returns the indices of a max pooling only together with the output
  md = (po("torch_ingress_num") %>>% nn("unsqueeze", dim = 2) %>>%
    nn("max_pool1d", kernel_size = 2, outputs = "indices"))$train(task)
  net = model_descriptor_to_module(md[[1L]])
  indices = net(torch_randn(2, 4))
  expect_equal(indices$dtype, torch_long())
  expect_equal(indices$shape, c(2, 1, 2))
})

test_that("$shapes_out() matches named output shapes by name", {
  po_two = R6Class("PipeOpTorchTwo", inherit = PipeOpTorch,
    public = list(initialize = function(id = "two") {
      super$initialize(id = id, module_generator = NULL, outname = c("a", "b"))
    }),
    private = list(.shapes_out = function(shapes_in, param_vals, task) {
      list(b = c(NA, 7L), a = shapes_in[[1L]])
    })
  )$new()
  expect_equal(po_two$shapes_out(list(c(NA, 3))), list(a = c(NA, 3L), b = c(NA, 7L)))
  po_two$outputs = "a"
  expect_equal(po_two$shapes_out(list(c(NA, 3))), list(a = c(NA, 3L)))
})

test_that("every PipeOpTorch includes $outputs in its hash", {
  # a subclass that overrides `.additional_phash_input()` without calling the parent's would lose it
  keys = grep("^nn_", mlr_pipeops$keys(), value = TRUE)
  required = list(nn_block = list(block = nn("linear", out_features = 1)), nn_fn = list(fn = identity))
  for (key in keys) {
    pipeop = invoke(po, key, .args = required[[key]])
    if (!inherits(pipeop, "PipeOpTorch")) next
    private = get_private(pipeop)
    # some operators leave out outputs by default, so start from all of them
    pipeop$outputs = private$.output_all$name
    phash = pipeop$phash
    # pretend that the module has another output, which is left out
    private$.output_all = rbind(private$.output_all,
      data.table(name = "__left_out__", train = "ModelDescriptor", predict = "Task"))
    expect_false(pipeop$phash == phash, info = key)
  }
})

test_that("the indices of a max pooling are only an output channel when requested", {
  expect_equal(nn("max_pool1d")$outputs, "output")
  expect_false(nn("max_pool1d")$hash == nn("max_pool1d", outputs = c("output", "indices"))$hash)
})

test_that("$outputs works for a block", {
  task = tsk("iris")
  block = gunion(list(nn("linear_1", out_features = 3), nn("linear_2", out_features = 5)))
  for (n_blocks in 0:1) {
    po_block = po("nn_block", block = block, n_blocks = n_blocks, outputs = "linear_2.output")
    expect_equal(po_block$output$name, "linear_2.output")
    graph = gunion(list(po("torch_ingress_num_1"), po("torch_ingress_num_2"))) %>>% po_block
    md = graph$train(task)
    expect_length(md, 1L)
    # with zero blocks, the block passes on its second input
    expect_equal(md[[1L]]$pointer_shape, if (n_blocks) c(NA, 5L) else c(NA, 4L))
  }
})

test_that("a block is not affected by changes to the graph it was constructed from", {
  block = as_graph(nn("max_pool1d", kernel_size = 1, outputs = c("output", "indices")))
  po_block = po("nn_block", block = block, n_blocks = 1)
  block$pipeops$max_pool1d$outputs = "indices"
  expect_equal(po_block$outputs, c("max_pool1d.output", "max_pool1d.indices"))
})

test_that("a block with zero repetitions needs as many inputs as outputs", {
  task = tsk("iris")
  block = as_graph(nn("max_pool1d", kernel_size = 1, outputs = c("output", "indices")))
  graph = po("torch_ingress_num") %>>% nn("unsqueeze", dim = 2) %>>%
    po("nn_block", block = block, n_blocks = 0, outputs = "max_pool1d.indices")
  expect_error(graph$train(task), "1 input\\(s\\) but 2 output\\(s\\)")
})
