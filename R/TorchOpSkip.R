# #' @title Adds a skip connection to a neural network
# #' @details
# #' @export
# TorchOpSkip = R6Class("TorchOpSkip",
#   inherit = TorchOp,
#   public = list(
#     initialize = function(id = "skip", param_vals = list(), .block) {
#       .block = assert_graph(.block)
#       private$.build = function(input, param_vals, task) {
#         .block = assert_graph(.block)
#         private$.build = function(input, param_vals, task) {
#           architecture = .block$train(task)[[2]]
#           block = reduce_architecture(architecture, task, input)
#           output_shape = with_no_grad(block(input))$shape
#           nn_module("skip",
#             initialize = function() {
#               self$block = block
#               self$linear = nn_linear()
#             },
#             forward = function(input) {
#               self$block(input) + self$linear(input)
#             }
#
#           )
#           return(layer_block)
#         }
#
#       }
#       param_set =
#       param_set = ps(
#         skip
#         block = p_uty(tags = "train")
#       )
#     }
#   ),
#   private = list(
#     .operator = "skip",
#     .build = NULL
#   )
# )
