#' @title Merge Operation
#'
#' @name mlr_pipeops_nn_merge
#' @template pipeop_torch_state_default
#'
#' @description
#' Base class for merge operations such as addition ([`PipeOpTorchMergeSum`]), multiplication
#' ([`PipeOpTorchMergeProd`] or concatenation ([`PipeOpTorchMergeCat`]).
#'
#' @section Input and Output Channels:
#' `PipeOpTorchMerge`s has either a *vararg* input channel if the constructor argument `innum` is not set, or
#' input channels `"input1"`, ..., `"input<innum>"`. There is one output channel `"output"`.
#' For an explanation see [`PipeOpTorch`].
#'
#' @section Parameters: See the respective child class.
#' @section Internals:
#' Per default, the `private$.shapes_out()` method outputs the broadcasted tensors. There are two things to be aware:
#' 1. `NA`s are assumed to batch (this should almost always be the batch size in the first dimension).
#' 2. Tensors are expected to have the same number of dimensions, i.e. missing dimensions are not filled with 1s.
#'    The reason is that again that the first dimension should be the batch dimension.
#' This private method can be overwritten by [`PipeOpTorch`]s inheriting from this class.
#'
#' @family PipeOps
#' @include PipeOpTorch.R
#' @export
PipeOpTorchMerge = R6Class("PipeOpTorchMerge",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @template param_module_generator
    #' @template param_param_set
    #' @param innum (`integer(1)`)\cr
    #'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
    initialize = function(id, module_generator, param_set = ps(), innum = 0, param_vals = list()) {
      private$.innum = assert_int(innum, lower = 0)
      inname = if (innum == 0) "..." else paste0("input", seq_len(innum))
      super$initialize(
        id = id,
        module_generator = module_generator,
        param_set = param_set,
        param_vals = param_vals,
        inname = inname,
        tags = "abstract"
      )
    }
  ),
  private = list(
    .innum = NULL,
    .shapes_out = function(shapes_in, param_vals, task) {
      # note that this slightly deviates from the actual broadcasting rules implemented by torch, i.e. we don't fill
      # up missing dimension with 1s because the first dimension is usually the batch dimension.
      assert_true(length(unique(map_int(shapes_in, length))) == 1)
      uniques = apply(as.data.frame(shapes_in), 1, function(row) {
        if (all(is.na(row))) {
          return(1)
        }
        max_dim = max(row, na.rm = TRUE)
        row[row == 1] = max_dim
        row = unique(row)
        sum(!is.na(row))
      })
      assert_true(all(uniques <= 1))
      shapes_in[1]
    }
  )
)

#' @title Merge by Summation
#'
#' @templateVar id nn_merge_sum
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit nn_merge_sum description
#'
#' @inheritSection mlr_pipeops_nn_merge Input and Output Channels
#' @section Parameters:
#' No parameters.
#' @section Internals:
#' Calls [`nn_merge_sum()`] when trained.
#' @family PipeOps
#' @export
PipeOpTorchMergeSum = R6Class("PipeOpTorchMergeSum", inherit = PipeOpTorchMerge,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param innum (`integer(1)`)\cr
    #'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
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


#' @title Merge by Product
#'
#' @templateVar id nn_merge_prod
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit nn_merge_prod description
#'
#' @inheritSection mlr_pipeops_nn_merge Input and Output Channels
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Internals:
#' Calls [`nn_merge_prod()`] when trained.
#'
#' @export
PipeOpTorchMergeProd = R6Class("PipeOpTorchMergeProd", inherit = PipeOpTorchMerge,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param innum (`integer(1)`)\cr
    #'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
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

#' @title Merge by Concatenation
#'
#' @templateVar id nn_merge_cat
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inherit nn_merge_cat description
#'
#' @inheritSection mlr_pipeops_nn_merge Input and Output Channels
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   The dimension along which to concatenate the tensors.
#' @section Internals:
#' Calls [`nn_merge_cat()`] when trained.
#' @export
PipeOpTorchMergeCat = R6Class("PipeOpTorchMergeCat", inherit = PipeOpTorchMerge,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @param innum (`integer(1)`)\cr
    #'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
    initialize = function(id = "nn_merge_cat", innum = 0, param_vals = list()) {
      param_set = ps(dim = p_int(default = -1, tags = "train"))
      super$initialize(
        id = id,
        module_generator = nn_merge_cat,
        innum = innum,
        param_set = param_set,
        param_vals = param_vals
      )
    },
    #' @description What does the cat say?
    speak = function() cat("I am the merge cat, meow! ^._.^\n")
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      assert_true(length(unique(map_int(shapes_in, length))) == 1)

      # dim can be negative (counting back from the last element which would be -1)
      true_dim = param_vals$dim %??% -1
      if (true_dim < 0) {
        true_dim = 1 + length(shapes_in[[1]]) + true_dim
      }
      assert_int(true_dim, lower = 1, upper = length(shapes_in[[1]]))

      shapes_matrix = as.matrix(as.data.frame(shapes_in))

      uniques = apply(shapes_matrix, 1, function(row) {
        if (all(is.na(row))) { # this is usually the batch dimension.
          return(1)
        }
        max_dim = max(row, na.rm = TRUE)
        row[row == 1] = max_dim
        row = unique(row)
        sum(!is.na(row))
      })

      # Dimensions don't have to match along the dimension along which we concatenate.
      assert_true(all(uniques[-true_dim] <= 1))

      returnshape = apply(shapes_matrix, 1, function(row) {
        row = unique(row)
        row = row[!is.na(row)]
        if (length(row)) max(row) else NA
      })
      returnshape[true_dim] = sum(shapes_matrix[true_dim, ])
      list(returnshape)
    }
  )
)

#' Product of multiple tensors
#'
#' Calculates the product of all input tensors.
#'
#' @export
nn_merge_prod = nn_module(
  "nn_merge_prod",
  initialize = function() NULL,
  forward = function(...) {
    Reduce(torch_mul, list(...))
  }
)

#' Sum of multiple tensors
#'
#' Calculates the sum of all input tensors.
#'
#' @export
nn_merge_sum = nn_module(
  "nn_merge_sum",
  initialize = function() NULL,
  forward = function(...) {
    Reduce(torch_add, list(...))
  }
)

#' Concatenates multiple tensors
#'
#' Concatenates multiple tensors on a given dimension.
#' No broadcasting rules are applied here, you must reshape the tensors before to have the same shape.
#'
#' @param dim (`integer(1)`)\cr
#'   The dimension for the concatenation.
#'
#' @export
nn_merge_cat = nn_module(
  "nn_merge_cat",
  initialize = function(dim = -1) self$dim = dim,
  forward = function(...) {
    torch_cat(list(...), dim = self$dim)
  }
)




#' @include zzz.R
register_po("nn_merge_sum", PipeOpTorchMergeSum)
register_po("nn_merge_prod", PipeOpTorchMergeProd)
register_po("nn_merge_cat", PipeOpTorchMergeCat)
