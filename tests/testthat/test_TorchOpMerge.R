test_that("mul and add works for valid input", {
  methods = c("mul", "add")
  for (method in methods) {
    task = tsk("mtcars")
    innum = 3L
    to = top("merge", method = "mul", .innum = innum)
    input = replicate(
      n = innum,
      torch_randn(10, 3)
    )
    input = set_names(input, nm = paste0("input", seq_len(innum)))
    y = torch_randn(10, 1)
    layer = to$build(input, task, y)
    out = do.call(layer$forward, input)
    expect_equal(out$shape, c(10L, 3L))
  }
})

test_that("mul and add fails for invalid input", {
  methods = c("mul", "add")
  for (method in methods) {
    task = tsk("mtcars")
    to = top("merge", method = "mul", .innum = 2L)
    input = list(
      input1 = torch_randn(10, 3),
      input2 = torch_randn(10, 4)
    )
    y = torch_randn(10, 1)
    expect_error(to$build(input, task, y))
  }
})

test_that("stack works for valid input", {
  task = tsk("mtcars")
  to1 = top("merge", method = "cat", dim = 1L, .innum = 2L)
  to2 = top("merge", method = "cat", dim = 2L, .innum = 2L)
  x = torch_randn(4, 7)
  input = list(
    input1 = x,
    input2 = x
  )
  layer1 = to1$build(input, task, NULL)
  layer2 = to2$build(input, task, NULL)
  out1 = do.call(layer1$forward, input)
  out2 = do.call(layer2$forward, input)
  expect_equal(out1$shape, c(8L, 7L))
  expect_equal(out2$shape, c(4L, 14L))
})

test_that("stack fails for invalid input", {
  task = tsk("mtcars")
  to = top("merge", method = "cat", dim = 1L, .innum = 2L)
  input = list(
    input1 = torch_randn(4, 7),
    input2 = torch_randn(4, 8)
  )
  expect_error(to$build(input, task, NULL))
})

test_that("works with vararg", {
  for (method in c("sum", "mul", "stack")) {
    x = torch_randn(4, 7)
    input = list(
      input1 = x,
      input2 = x
    )
    to = top("merge", method = "cat")
    layer = to$build(input, task, NULL)
    expect_error(do.call(layer, args = input), regexp = NA)
  }
})
