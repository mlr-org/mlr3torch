test_that("merging works", {
  g1 = as_graph(po("nop_1"))
  g2 = po("nop_1") %>>% po("nop_2")

  # mer
  g3 = merge_graphs(g1, g2)

  expect_equal(g2$edges, g3$edges)
  expect_equal(names(g2$pipeops), names(g3$pipeops))
})

test_that("merge_lazy_tensor_graphs works", {
  lt1 = as_lazy_tensor(torch_randn(10, 3))
  y = runif(10)

  task = as_task_regr(data.table(lt = lt1, y = y), target = "y")

  graph = po("nop") %>>%
    po("lazy_transform", function(x) x * 100) %>>%
    list(po("lazy_transform_1", function(x) x / 100), po("lazy_transform_2", function(x) x))

  res = graph$train(task)

  c1 = res[[1L]]$data(cols = "lt")[[1]]
  c2 = res[[2L]]$data(cols = "lt")[[1]]

  g = merge_lazy_tensor_graphs(data.table(c1 = c1, c2 = c2))
})

