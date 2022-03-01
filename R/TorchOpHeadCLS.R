# TorchOpHeadCLS = R6Class("TorchOpHeadCLS",
#   public = list(
#     initialize = function() {
#       param_set = ps(
#         activation = p_uty(tags = "train") ,
#         normaliztion = p_uty(tags = "train")
#       )
#       super$initialize(
#         id = id,
#         param_set = param_set,
#         param_vals = param_vals
#       )
#     }
#   ),
#   private = list(
#     .build = function(input, param_vals, task) {
#       d_token = input$shape[[3]]
#
#     }
#   )
# )
#
# nn_head_cls = nn_module("nn_head_cls",
#   initialize = function(normalization, activation) {
#     self$
#   }
# )
#
#    class Head(nn.Module):
#         """The final module of the `Transformer` that performs BERT-like inference."""
#
#         def __init__(
#             self,
#             *,
#             d_in: int,
#             bias: bool,
#             activation: ModuleType,
#             normalization: ModuleType,
#             d_out: int,
#         ):
#             super().__init__()
#             self.normalization = _make_nn_module(normalization, d_in)
#             self.activation = _make_nn_module(activation)
#             self.linear = nn.Linear(d_in, d_out, bias)
#
#         def forward(self, x: Tensor) -> Tensor:
#             x = x[:, -1]
#             x = self.normalization(x)
#             x = self.activation(x)
#             x = self.linear(x)
#             return x
#
#
#
