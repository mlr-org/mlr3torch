# #' @title Autotest for TorchOps
# #'
# #' @description This function tests whether a [TorchOp] satisfies some minimal conditions and is
# #' used for testing purposes
# #' @param op (`TorchOp`)\cr
# #'   The TorchOp to be tested.
# #' @param search_space (`TorchOp`)\cr
# #'   The search_space used to generate parameters. This is used to check whether the parameters
# #'   are having an effect (not whether they are passed correctly).
# #' @export
# autotest_torchop = function(op) {
#   cls = getFromNamespace(class(op)[[1L]], ns = getNamespace("mlr3torch"))
#
#   # until we really need it
#   expect_true(op$outnum == 1L)
#
# }
