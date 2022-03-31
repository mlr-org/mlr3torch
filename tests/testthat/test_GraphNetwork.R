test_that("Linear GraphNetwork works", {
  batch_size = 16L
  d_token = 3L
  task = tsk("iris")
  batch = make_batch(task, batch_size = batch_size)
  graph = top("tokenizer", d_token = d_token) %>>%
    top("flatten") %>>%
    top("linear1", out_features = 10L) %>>%
    top("relu1") %>>%
    top("linear2", out_features = 1L)
  architecture = graph$train(task)[[1L]][[2L]]
  net = architecture$build(task)
  y_hat = net$forward(batch)
  expect_equal(y_hat$shape, c(batch_size, 1L))
})

test_that("GraphNetwork with fork (depth 1) works", {
  d_token = 4L
  batch_size = 9L
  task = tsk("iris")
  batch = make_batch(task, batch_size)
  graph = top("tokenizer", d_token = d_token) %>>%
    top("flatten") %>>%
    # top("fork", .outnum = 2L) %>>%
    gunion(
      graphs = list(
        a = top("linear", out_features = 3L) %>>% top("relu"),
        b = top("linear", out_features = 3L)
      )
    ) %>>%
    top("merge", method = "add", .innum = 2L) %>>%
    top("linear", out_features = 1L)
  architecture = graph$train(task)[[1L]][[2L]]
  net = architecture$build(task)
  y_hat = net$forward(batch)
  expect_equal(y_hat$shape, c(batch_size, 1L))
})

test_that("GraphNetwork with fork (depth 2) works", {
  #
  #                                  --> aa.linear -->
  #                      --> a.linear
  # tokenizer --> flatten            --> ab.linear --> merge
  #
  #                      --> b.linear --------------->
  d_token = 4L
  batch_size = 9L
  task = tsk("iris")
  batch = make_batch(task, batch_size)
  a = top("fork2", .outnum = 2L) %>>%
    gunion(
      graphs = list(
        c = top("linear", out_features = 3L),
        d = top("linear", out_features = 3L)
      )
    ) %>>%
    top("merge", method = "mul", .innum = 2L)


  graph = top("tokenizer", d_token = d_token) %>>%
    top("flatten") %>>%
    top("fork1", .outnum = 2L) %>>%
    gunion(
      graphs = list(
        a = a,
        b = top("linear", out_features = 3L)
      )
    ) %>>%
    top("merge", method = "add", .innum = 2L) %>>%
    top("linear", out_features = 1L)
  architecture = graph$train(task)[[1L]][[2L]]
  net = architecture$build(task)
  y_hat = net$forward(batch)
  expect_equal(y_hat$shape, c(batch_size, 1L))
})

test_that("Another fork", {
  task = tsk("iris")
  batch = get_batch(task)
  graph = top("tokenizer", d_token = 2L) %>>%
    top("fork")

})
