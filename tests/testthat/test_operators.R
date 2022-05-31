test_that("+ works", {
  p1 = top("linear_1", out_features = 10)
  p2 = top("linear_2", out_features = 10)

  g = p1 %++% p2
  expect_r6(g, "Graph")
  expect_true(g$rhs == "add")
  expect_true(all(g$lhs == c("linear_1", "linear_2")))

  p3 = top("linear_3", out_features = 10)

  g = g %++% p3
  expect_r6(g, "Graph")
  expect_true(g$rhs == "add_1")
})
