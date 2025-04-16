test_that("linear graph", {
  block = po("nn_linear", out_features = 10) %>>% po("nn_relu")

  po_block = po("nn_block", block, n_blocks = 2)
  expect_pipeop(po_block)

  comp_graph = po("nn_linear", out_features = 10L, id = "nn_linear__1") %>>%
    po("nn_relu", id = "nn_relu__1") %>>%
    po("nn_linear", out_features = 10L, id = "nn_linear__2") %>>%
    po("nn_relu", id = "nn_relu__2")

  comp_graph$update_ids(prefix = "nn_block.")

  task = tsk("iris")

  md1 = po("torch_ingress_num")$train(list(task))

  md2 = po("torch_ingress_num")$train(list(task))

  gblock = po_block$train(md1)[[1L]]$graph
  gcomp = comp_graph$train(md2[[1L]], single_input = TRUE)[[1L]]$graph

  expect_equal(
    gblock$ids(sorted = TRUE),
    gcomp$ids(sorted = TRUE)
  )
  expect_equal(
    gblock$edges,
    gcomp$edges
  )
})

test_that("deep clone works", {
  block = po("nn_linear", out_features = 10) %>>% po("nn_relu")
  po_block = po("nn_block", block, n_blocks = 2)
  po_block_c = po_block$clone(deep = TRUE)
  expect_deep_clone(po_block, po_block_c)

  # check that parameters are still passed correctly
  po_block_c$param_set$set_values(
    nn_linear.out_features = 20L,
    n_blocks = 2L
  )
  task = tsk("iris")
  md = po("torch_ingress_num")$train(list(task))
  mdout = po_block_c$train(md)
  expect_equal(mdout[[1L]]$pointer_shape, c(NA, 20L))
  expect_equal(sum(startsWith(mdout[[1L]]$graph$ids(), "nn_block.nn_linear")), 2L)
})

test_that("shapes_out works", {
  block = list(po("nn_linear_1", out_features = 1), po("nn_linear_2", out_features = 2)) %>>% po("nn_merge_cat")
  po_block = po("nn_block", block, n_blocks = 2)

  task = tsk("iris")
  res = po_block$shapes_out(list(c(NA, 4), c(NA, 4)), task = task)
  expect_error(
    po_block$shapes_out(list(c(NA, 4), c(NA, 4))),
    "requires a task"
  )
})

test_that("works when including non-torch pipeop", {
  task = tsk("iris")
  block = ppl("branch", list(nn_relu = po("nn_relu"), nn_linear = po("nn_linear", out_features = 2L)))
  po_block = po("nn_block", block, n_blocks = 1L)
  po_block$param_set$set_values(
    branch.selection = "nn_relu"
  )
  expect_equal(
    po_block$shapes_out(list(c(NA, 4)), task)[[1L]],
    c(NA, 4)
  )
  po_block$param_set$set_values(
    branch.selection = "nn_linear"
  )
  expect_equal(
    po_block$shapes_out(list(c(NA, 4)), task)[[1L]],
    c(NA, 2L)
  )
  md = po("torch_ingress_num")$train(list(task))[[1L]]
  mdout = po_block$train(list(md))[[1L]]
  expect_false("nn_block.nn_relu__1" %in% mdout$graph$ids())
  expect_true("nn_block.nn_linear__1" %in% mdout$graph$ids())
})

test_that("different block changes phash", {
  x1 = po("nn_block", po("nn_linear"))
  x2 = po("nn_block", po("nn_relu"))
  expect_false(x1$phash == x2$phash)
})

test_that("0 blocks are possible", {
  md = po("torch_ingress_num")$train(list(tsk("iris")))[[1L]]
  mdout = nn("block", block = nn("linear", out_features = 10), n_blocks = 0)$train(list(md))[[1L]]
  expect_equal(mdout$pointer_shape, c(NA, 4))
})