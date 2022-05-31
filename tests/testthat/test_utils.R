test_that("split_list works", {
  l = list(1, b = 2, 3, 4, a = 7)
  observed = split_list(l, c("c", "a", "b"))
  expected = list(named_list(), list(a = 7), list(b = 2))
  expect_equal(expected, observed)

  observed = split_list(l, c("a|^$", "b"))
  expected = list(list(1, 3, 4, a = 7), list(b = 2))
  expect_equal(expected, observed)

  l = list(a = 1, b = 2)
  observed = split_list(l, c(aaa = "a"))
  expected = list(aaa = list(a = 1))
  expect_equal(observed, expected)

  l = list(1, 2)
  observed = split_list(list(1, 2), c("*"))
  expected = list(list(1, 2))
  expect_equal(observed, expected)

  expect_error(split_list(l, list("a", "b")))

  l = list()
  observed = split_list(l, c("a", "b"))
  expected = list(list(), list())
  expect_equal(observed, expected)
})
