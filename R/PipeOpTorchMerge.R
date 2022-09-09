#' @title Merges Multiple Tensors
#'
#' @description
#' Merges multiple tensors.
#'
#' @section Order of inputs:
#' In case the order of input matters (e.g. for method "cat"), the constructor argument 'innum'
#' should be set.
#' @export
#'
#' @examples
#' @rdname torchop_merge
PipeOpTorchMerge = R6Class("PipeOpTorchMerge",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the new object.
    #' @param param_set (`paradox::ParamSet`)\cr
    #'   The parameter set.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param innum (`integer(1)`)\cr
    #'   The number of input channels (optional). If provided, the input channels are set to
    #'   `"input1"`, `"input2"`, etc.. Otherwise the input channel is set to `...` (a 'vararg' channel).
    #'   Should be set, in case the order of the inputs is relevant.
    #' @param .method (`character(1)`)\cr
    #'   The method for the concatenation. One of "add", "mul" or "cat".
    initialize = function(id, module_generator, param_set = ps(), innum = 0, param_vals = list()) {
      super$initialize(
        id = id,
        module_generator = module_generator,
        param_set = param_set,
        param_vals = param_vals,
        multi_input = innum
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      assert_true(length(unique(map_int(shapes_in, length))) == 1)
      uniques = apply(as.data.frame(shapes_in), 1, function(row) {
        row = unique(row)
        sum(!is.na(row))
      })
      assert_true(all(uniques <= 1))
      shapes_in[1]
    }
  )
)

#' @rdname torchop_merge
#' @export
PipeOpTorchMergeSum = R6Class("PipeOpTorchMergeSum", inherit = PipeOpTorchMerge,
  public = list(
    initialize = function(id = "nn_merge_sum", innum = 0, param_vals = list()) {
      super$initialize(
        id = id,
        module_generator = nn_merge_sum,
        innum = innum,
        param_vals = param_vals
      )
    }
  )
)

#' @rdname torchop_merge
#' @export
PipeOpTorchMergeProd = R6Class("PipeOpTorchMergeProd", inherit = PipeOpTorchMerge,
  public = list(
    initialize = function(id = "nn_merge_prod", innum = 0, param_vals = list()) {
      super$initialize(
        id = id,
        module_generator = nn_merge_prod,
        innum = innum,
        param_vals = param_vals
      )
    }
  )
)

#' @rdname torchop_merge
#' @export
PipeOpTorchMergeCat = R6Class("PipeOpTorchMergeCat", inherit = PipeOpTorchMerge,
  public = list(
    initialize = function(id = "nn_merge_cat", innum = 0, param_vals = list()) {
      param_set = ps(dim = p_int(tags = c("train", "required")))
      param_set$values$dim = -1
      super$initialize(
        id = id,
        module_generator = nn_merge_cat,
        innum = innum,
        param_set = param_set,
        param_vals = param_vals
      )
    },
    speak = function() cat("I am the merge cat, meow! ^._.^\n")
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) {
      assert_true(length(unique(map_int(shapes_in, length))) == 1)

      true_dim = param_vals$dim
      if (true_dim < 0) {
        true_dim = 1 + length(shapes_in[[1]]) + true_dim
      }
      assert_int(true_dim, lower = 1, upper = length(shapes_in[[1]]))

      shapes_matrix = as.matrix(as.data.frame(shapes_in))

      uniques = apply(shapes_matrix, 1, function(row) {
        row = unique(row)
        sum(!is.na(row))
      })

      assert_true(all(uniques[-true_dim] <= 1))

      returnshape = apply(shapes_matrix, 1, function(row) {
        row = unique(row)
        row = row[!is.na(row)]
        if (length(row)) row[[1]] else NA
      })
      returnshape[true_dim] = sum(shapes_matrix[true_dim, ])
      list(returnshape)
    }
  )
)


nn_merge_prod = nn_module(
  "nn_merge_prod",
  initialize = function() NULL,
  forward = function(...) {
    torch_prod(torch_stack(list(...)), dim = 1L)
  }
)

nn_merge_sum = nn_module(
  "nn_merge_sum",
  initialize = function() NULL,
  forward = function(...) {
    torch_sum(torch_stack(list(...)), dim = 1L)
  }
)

nn_merge_cat = nn_module(
  "nn_merge_cat",
  initialize = function(dim) self$dim = dim,
  forward = function(...) {
    torch_cat(list(...), dim = self$dim)
  }
)

#' @include zzz.R
register_po("nn_merge_sum", PipeOpTorchMergeSum)
register_po("nn_merge_prod", PipeOpTorchMergeProd)
register_po("nn_merge_cat", PipeOpTorchMergeCat)
