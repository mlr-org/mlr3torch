test_that("+ works", {
  p1 = top("linear_1", out_features = 10)
  p2 = top("linear_2", out_features = 10)

  g = p1 %++% p2
  expect_r6(g, "Graph")
  expect_true(g$rhs == "add")
  expect_true(all(g$lhs == c("linear_1", "linear_2")))

  g = p1 %**% p2
  expect_r6(g, "Graph")
  expect_true(g$rhs == "mul")
  expect_true(all(g$lhs == c("linear_1", "linear_2")))

  g = p1 %cc% p2
  expect_r6(g, "Graph")
  expect_true(g$rhs == "cat")
  expect_true(all(g$lhs == c("linear_1", "linear_2")))
})
