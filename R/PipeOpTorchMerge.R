#' @title Merge Operation
#'
#' @usage NULL
#' @name mlr_pipeops_torch_merge
#' @format `r roxy_pipeop_torch_format()`
#'
#' @description
#' Base class for merge operations.
#'
#' @section Construction: `r roxy_construction(PipeOpTorchMerge)`
#' * `r roxy_param_id("module")`
#' * `r roxy_param_param_vals()`
#' * `r roxy_param_param_set()`
#' * `r roxy_param_module_generator()`
#' * `innum` :: `integer(1)`\cr
#'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
#'
#' @section Input and Output Channels:
#' `PipeOpTorchMerge`s has either a *vararg* input channel if the constructor argument `innum` is not set, or
#' input channels `"input1"`, ..., `"input<innum>"`. There is one output channel `"output"`.
#' For an explanation see [`PipeOpTorch`].
#'
#' @section State: `r roxy_pipeop_torch_state_default()`
#' @section Parameters: See the respective child class.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' See the respective child class.
#' @family PipeOpTorch
#' @export
PipeOpTorchMerge = R6Class("PipeOpTorchMerge",
  inherit = PipeOpTorch,
  public = list(
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

#' @title Merge by Summation
#'
#' @usage NULL
#' @name mlr_pipeops_torch_merge_sum
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit nn_merge_sum description
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchMergeSum)`
#'
#' * `r roxy_param_id("nn_merge_sum")`
#' * `r roxy_param_param_vals()`
#' * `innum` :: `integer(1)`\cr
#'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
#'
#' @section Input and Output Channels:
#' `r roxy_pipeop_torch_channels_default()`
#'
#' @section State:
#' `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_merge_sum()`] when trained.
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_merge_sum")
#' obj$id
#' obj$module_generator
#' obj$shapes_out(list(input1 = c(16, 5, 5), input2 = c(16, 5, 5)))
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


#' @title Merge by Product
#'
#' @usage NULL
#' @name mlr_pipeops_torch_merge_prod
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit nn_merge_prod description
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchMergeProd)`
#' * `r roxy_param_id("nn_merge_prod")`
#' * `r roxy_param_param_vals()`
#' * `innum` :: `integer(1)`\cr
#'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_merge_prod()`] when trained.
#'
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_merge_prod")
#' obj$id
#' obj$module_generator
#' obj$shapes_out(list(input1 = c(16, 5, 5), input2 = c(16, 5, 5)))
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

#' @title Merge by Concatenation
#'
#' @usage NULL
#' @name mlr_pipeops_torch_merge_cat
#' @format `r roxy_pipeop_torch_format()`
#'
#' @inherit nn_merge_cat description
#'
#' @section Construction:
#' ```
#' PipeOpTorchMergeCat$new(id = "nn_merge_prod", innum = 0, param_vals = list())
#' ```
#' * `r roxy_param_id("nn_merge_cat")`
#' * `r roxy_param_param_vals()`
#' * `innum` :: `integer(1)`\cr
#'   The number of inputs. Default is 0 which means there is one *vararg* input channel.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State: `r roxy_pipeop_torch_state_default()`
#'
#' @section Parameters:
#' No parameters.
#'
#' @section Fields: `r roxy_pipeop_torch_fields_default()`
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#' @section Internals:
#' Calls [`nn_merge_cat()`] when trained.
#'
#' @family PipeOpTorch
#' @export
#' @examples
#' obj = po("nn_merge_cat", dim = 2)
#' obj$id
#' obj$module_generator
#' obj$shapes_out(list(input1 = c(16, 5, 7), input2 = c(16, 6, 7)))
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
    #' @description What does the cat say?
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

#' Product of multiple tensors
#'
#' Calculates the product of all input tensors.
#'
#' @export
nn_merge_prod = nn_module(
  "nn_merge_prod",
  initialize = function() NULL,
  forward = function(...) {
    torch_prod(torch_stack(list(...)), dim = 1L)
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
    torch_sum(torch_stack(list(...)), dim = 1L)
  }
)

#' Concatenates multiple tensors
#'
#' Concatenates multiple tensors on a given dimension.
#'
#' @param dim (`integer(1)`)\cr
#'   The dimension for the concatenation.
#'
#' @export
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
