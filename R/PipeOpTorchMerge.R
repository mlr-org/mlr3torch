#' @title Merge Operation
#' @description
#' Base class for merge operations such as addition ([`PipeOpTorchMergeSum`]), multiplication
#' ([`PipeOpTorchMergeProd`] or concatenation ([`PipeOpTorchMergeCat`]).
#' @section Parameters: See the respective child class.
#'
#' @name mlr_pipeops_nn_merge
#' @template pipeop_torch_state_default
#' @section Input and Output Channels:
#' `PipeOpTorchMerge`s has either a *vararg* input channel if the constructor argument `innum` is not set, or
#' input channels `"input1"`, ..., `"input<innum>"`. There is one output channel `"output"`.
#' For an explanation see [`PipeOpTorch`].
#'
#' @section Internals:
#' Per default, the `private$.shapes_out()` method outputs the broadcasted tensors. There are two things to be aware:
#' 1. `NA`s are assumed to batch (this should almost always be the batch size in the first dimension).
#' 2. Tensors are expected to have the same number of dimensions, i.e. missing dimensions are not filled with 1s.
#'    The reason is again that the first dimension should be the batch dimension.
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
    .additional_phash_input = function() {
      list(private$.innum)
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      # note that this slightly deviates from the actual broadcasting rules implemented by torch, i.e. we don't fill
      # up missing dimension with 1s because the first dimension is usually the batch dimension.
      assert_same_ndim(shapes_in, self$id)
      # the output is the broadcast of the inputs, not the first input: broadcasting a (NA, 1)
      # against a (NA, 6) yields a (NA, 6) tensor at runtime
      list(broadcast_shapes(shapes_in, self$id))
    }
  )
)

#' @title Merge by Summation
#'
#' @inherit nn_merge_sum description
#' @section nn_module:
#' Calls [`nn_merge_sum()`] when trained.
#' @section Parameters:
#' No parameters.
#' @templateVar id nn_merge_sum
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @inheritSection mlr_pipeops_nn_merge Input and Output Channels
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
#' @inherit nn_merge_prod description
#' @section nn_module:
#' Calls [`nn_merge_prod()`] when trained.
#' @section Parameters:
#' No parameters.
#'
#' @templateVar id nn_merge_prod
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @inheritSection mlr_pipeops_nn_merge Input and Output Channels
#'
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
#' @inherit nn_merge_cat description
#' @section nn_module:
#' Calls [`nn_merge_cat()`] when trained.
#' @section Parameters:
#' * `dim` :: `integer(1)`\cr
#'   The dimension along which to concatenate the tensors. The default is -1, i.e., the last dimension.
#' @templateVar id nn_merge_cat
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#'
#' @inheritSection mlr_pipeops_nn_merge Input and Output Channels
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
      assert_same_ndim(shapes_in, self$id)

      dim = param_vals[["dim"]] %??% -1L
      true_dim = resolve_dim(dim, shapes_in[[1L]])
      assert_dim_in_range(dim, true_dim, shapes_in[[1L]], self$id)

      # `torch_cat()` does not broadcast: every dimension except the concatenated one must be
      # equal, which is checked here rather than at runtime
      returnshape = cat_shapes(shapes_in, true_dim, self$id)
      # the sizes along the concatenated dimension are summed, and are unknown as soon as one is
      returnshape[true_dim] = sum(map_dbl(shapes_in, function(shape) shape[true_dim]))
      list(as.integer(returnshape))
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

#' @include aaa.R
register_po("nn_merge_sum", PipeOpTorchMergeSum)
register_po("nn_merge_prod", PipeOpTorchMergeProd)
register_po("nn_merge_cat", PipeOpTorchMergeCat)
