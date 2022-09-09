#' @title Base Class for Torch Module Wrappers
#'
#' @description
#' `PipeOpModule` wraps a `torch::nn_module`. Graphs of `PipeOpModule`s are combined in a [`GraphModule`].
#'
#' @examples
#' @export
PipeOpModule = R6Class("PipeOpModule",
  inherit = PipeOp,
  public = list(
    module = NULL,
    #' @description Initializes a new instance of this [R6 Class][R6::R6Class].
    #' @param multi_input (`NULL` | `integer(1)`)\cr
    #'   `0`: `...`-input. Otherwise: `multi_input` times input channel named `input1:`...`input#`.\cr
    #'   `module`'s `$forward` function must take `...`-input if `multi_input` is 0, and must have `multi_input` arguments otherwise.
    #' @param multi_output (`NULL` | `integer(1)`)\cr
    #'   `NULL`: single output. Otherwise: `multi_output` times output channel named `output1:`...`input#`.\cr
    #'   `module`'s `$forward` function must return a `list` of `torch_tensor` if `multi_output` is not `NULL`.
    initialize = function(id, module, multi_input = 1, multi_output = NULL, param_vals = list(), packages = character(0)) {
      # default input and output channels, packages
      assert_int(multi_input, null.ok = FALSE, lower = 0)
      assert_int(multi_output, null.ok = TRUE, lower = 1)
      self$module = assert_class(module, "nn_module")
      assert_character(packages, any.missing = FALSE)

      inname = if (multi_input == 0) "..." else sprintf("input%s", seq_len(multi_input))
      outname = if (is.null(multi_output)) "output" else sprintf("output%s", seq_len(multi_output))
      input = data.table(name = inname, train = "torch_tensor", predict = "NULL")
      output = data.table(name = outname, train = "torch_tensor", predict = "NULL")

      super$initialize(
        id = id,
        input = input,
        output = output,
        param_vals = param_vals,
        packages = packages
      )
    }
    # TODO: printer that calls the nn_module's printer
    # TODO: maybe call input just 'input' and not 'input1' if only one input present
    # TODO: make module a read-only active binding
  ),
  private = list(
    .train = function(inputs) {
      self$state = list()  # PipeOp API requires this.
      outputs = do.call(self$module, unname(inputs))
      outname = self$output$name
      if (identical(outname, "output")) outputs = list(outputs)  # the only case where module does not produce a list
      outputs
    },
    .predict = function(inputs) {
      rep(list(NULL), nrow(self$output))
    }
  )
)

