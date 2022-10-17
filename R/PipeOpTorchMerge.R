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
#' @template param_id
#' @template param_param_vals
#' @template param_param_set
#' @param innum (`integer(1)`)\cr
#'   The number of input channels (optional). If provided, the input channels are set to
#'   `"input1"`, `"input2"`, etc.. Otherwise the input channel is set to `...` (a 'vararg' channel).
#'   Should be set, in case the order of the inputs is relevant.
#'
#' @examples
#' @rdname torchop_merge
PipeOpTorchMerge = R6Class("PipeOpTorchMerge",
  inherit = PipeOpTorch,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    initialize = function(id, module_generator, param_set = ps(), innum = 0, param_vals = list()) {
      private$.innum = assert_int(innum, lower = 0)
      inname = if (innum == 0) "..." else paste0("input", seq_len(innum))
      super$initialize(
        id = id,
        module_generator = module_generator,
        param_set = param_set,
        param_vals = param_vals,
        inname = inname
      )
    }
  ),
  private = list(
    .innum = NULL,
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
      private$.innum = innum
      super$initialize(
        id = id,
        module_generator = NULL,
        innum = innum,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .innum = NULL,
    .make_module = function(shapes_in, param_vals) {
      innum = private$.innum
      argnames = if (innum == 0) "..." else paste0(paste0("input", seq_len(innum), collapse = ", "))
      call = sprintf("torch_sum(torch_stack(list(%s)), dim = 1L)", argnames)
      forward = eval(str2lang(sprintf("function(%s) %s", argnames, call)))
      nn_module("nn_merge_sum", forward = forward)()
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
  ),
  private = list(
    .make_module = function(shapes_in, param_vals) {
      innum = private$.innum
      argnames = if (innum == 0) "..." else paste0(paste0("input", seq_len(innum), collapse = ", "))
      call = sprintf("torch_prod(torch_stack(list(%s)), dim = 1L)", argnames)
      forward = eval(str2lang(sprintf("function(%s) %s", argnames, call)))
      nn_module("nn_merge_prod", forward = forward)()
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
    .make_module = function(shapes_in, param_vals) {
      innum = private$.innum
      argnames = if (innum == 0) "..." else paste0(paste0("input", seq_len(innum), collapse = ", "))
      call = sprintf("torch_cat(list(%s)), dim = self$dim)", argnames)
      forward = eval(str2lang(sprintf("function(%s) %s", argnames, call)))
      nn_module("nn_merge_cat", forward = forward)()
    },
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

# nn_merge_cat = nn_module("nn_merge_cat",
#   NULL
#
# )


#' @include zzz.R
register_po("nn_merge_sum", PipeOpTorchMergeSum)
register_po("nn_merge_prod", PipeOpTorchMergeProd)
register_po("nn_merge_cat", PipeOpTorchMergeCat)
