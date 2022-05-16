#' @export
#' What do we want from the autotest:
#' What needs to be specified in the TorchOp for the autotest to run?
test_torchop = function(torchop, ...) {
  UseMethod("test_torchop")
}

test_torchop.TorchOp = function(torchop, ...) {
  # We test the following:
  # 1. Can be constructed using the dictionary
  torchop_sugar = top(get_private(torchop)$.operator)
  expect_equal(torchop$new(), torchop_sugar)

  # We need to create a minimally working architecture and see whether
  # 1. It runs
  # 2. The parameters are correct
  # 3. It is constructed correctöly

}

test_torchop.default = function(torchop, ...) {
  stopf("Cannot test object of class %s.", class(torchop)[[1L]])
}
