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
#' library("mlr3pipelines")
#'
#' top1 = top("linear_1", out_features = 10L)
#' top2 = top("linear_2", out_features = 10L)
#'
#' # order is not specified:
#' top_add = top("add")
#' graph = gunion(list(top1, top2)) %>>% top_add
#'
#' # order is specified:
#' top_cat = top("cat", .nnum = 2)
#' graph = gunion(list(top1, top2))
#' graph$add_pipeop(top_cat)
#' graph$add_edge(src_id = "linear_1", dst_id = "cat", dst_channel = "input2")
#' graph$add_edge(src_id = "linear_2", dst_id = "cat", dst_channel = "input1")
#' @rdname torchop_merge
TorchOpMerge = R6Class("TorchOpMerge",
  inherit = TorchOp,
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
    initialize = function(id = "merge", param_set = ps(), param_vals = list(), .method,
      innum = NULL) {
      private$.method = assert_choice(.method, c("add", "mul", "cat"))
      if (is.null(innum)) {
        input = data.table(
          name = "...",
          train = "ModelArgs",
          predict = "Task"
        )
      } else {
        input = data.table(
          name = paste0("input", seq_len(innum)),
          train = rep("ModelArgs", times = innum),
          predict = rep("Task", times = innum)
        )
      }
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input
      )
    }
  ),
  private = list(
    .build = function(inputs, task) {
      pv = self$param_set$get_values(tag = "train")

      fn = switch(private$.method,
        add = nn_merge_sum,
        mul = nn_merge_mul,
        cat = nn_merge_cat
      )
      if (private$.method == "cat") {
        args = list(dim = self$param_set$values$dim)
      } else {
        args = list()
      }
      invoke(fn, .args = args)
    },
    .method = NULL
  )
)

#' @rdname torchop_merge
#' @export
TorchOpAdd = R6Class("TorchOpAdd",
  inherit = TorchOpMerge,
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
    initialize = function(id = "add", param_vals = list(), innum = NULL) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .method = "add",
        innum = innum
      )
    }
  )
)

#' @rdname torchop_merge
#' @export
TorchOpMul = R6Class("TorchOpMul",
  inherit = TorchOpMerge,
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
    initialize = function(id = "mul", param_vals = list(), innum = NULL) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .method = "mul",
        innum = innum
      )
    }
  )
)

#' @rdname torchop_merge
#' @export
TorchOpCat = R6Class("TorchOpCat",
  inherit = TorchOpMerge,
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
    initialize = function(id = "cat", param_vals = list(), innum = NULL) {
      param_set = ps(
        dim = p_uty(tags = c("train", "required"), custom_check = check_cat_dim)
      )
      super$initialize(
        id = id,
        param_vals = param_vals,
        param_set = param_set,
        .method = "cat",
        innum = innum
      )
    }
  )
)

check_cat_dim = function(x) {
  msg = check_int(x, lower = 1L)
  if (is.character(msg)) {
    return(msg)
  }
  if (x == 1L) {
    warningf("The parameter 'dim' of TorchOpCat is set to 1L, which is probably the batch dimension")
  }
  TRUE
}

nn_merge_mul = nn_module(
  "merge_multiply",
  initialize = function() NULL,
  forward = function(...) {
    torch_prod(torch_stack(list(...)), dim = 1L)
  }
)

nn_merge_sum = nn_module(
  "merge_sum",
  initialize = function() NULL,
  forward = function(...) {
    torch_sum(torch_stack(list(...)), dim = 1L)
  }
)

nn_merge_cat = nn_module(
  "merge_cat",
  initialize = function(dim) self$dim = dim,
  forward = function(...) {
    torch_cat(list(...), dim = self$dim)
  }
)

#' @include mlr_torchops.R
mlr_torchops$add("merge", TorchOpMerge)
#' @include mlr_torchops.R
mlr_torchops$add("add", TorchOpAdd)
#' @include mlr_torchops.R
mlr_torchops$add("mul", TorchOpMul)
#' @include mlr_torchops.R
mlr_torchops$add("cat", TorchOpCat)
