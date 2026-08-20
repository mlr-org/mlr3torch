# Base class for the embedding lookups. They share the table itself -- `num_embeddings` rows of
# `embedding_dim` values, indexed by the input -- and differ in what they do with the result.
PipeOpTorchEmbeddingBase = R6Class("PipeOpTorchEmbeddingBase",
  inherit = PipeOpTorch,
  public = list(
    # @description
    # Creates a new instance of this [R6][R6::R6Class] class.
    # @template params_pipelines
    # @template param_module_generator
    # @template param_param_set
    initialize = function(id, module_generator, param_set, param_vals = list()) {
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = module_generator
      )
    }
  ),
  private = list(
    # The size of the embedding table. It is the `num_embeddings` parameter when the user set it,
    # and otherwise the largest number of categories over the task's categorical features, so that
    # every code the ingress produces has a row. `NULL` when neither source is available.
    .num_embeddings = function(param_vals, task) {
      if (!is.null(param_vals[["num_embeddings"]])) {
        return(as.integer(param_vals[["num_embeddings"]]))
      }
      if (is.null(task)) {
        return(NULL)
      }
      cardinalities = categ_cardinalities(task)
      if (!length(cardinalities)) {
        return(NULL)
      }
      max(cardinalities)
    },
    .assert_num_embeddings = function(param_vals, task) {
      num_embeddings = private$.num_embeddings(param_vals, task)
      # the table cannot be sized without this, so it is a configuration error rather than a shape
      # that is merely not known yet
      if (is.null(num_embeddings)) {
        stopf("PipeOp '%s' needs to know the number of rows of the embedding table, but the parameter 'num_embeddings' is not set and the task has no categorical features to infer it from.", # nolint
          self$id)
      }
      num_embeddings
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$num_embeddings = private$.assert_num_embeddings(param_vals, task)
      param_vals
    }
  )
)

#' @title Embedding
#' @inherit torch::nn_embedding description
#' @section nn_module:
#' Calls [`torch::nn_embedding()`] when trained.
#'
#' The input is a tensor of indices, e.g. the output of [`PipeOpTorchIngressCategorical`], and the
#' output has one more dimension than the input, namely `embedding_dim`.
#' The indices start at 1, as they do everywhere in `torch`'s R interface.
#'
#' Note that a single table is shared by all positions of the input, so applying this to the
#' categorical features of a task embeds two features that happen to have the same code into the
#' same vector. Use [`nn("tokenizer_categ")`][mlr_pipeops_nn_tokenizer_categ] for one table per
#' feature.
#'
#' @section Parameters:
#' * `embedding_dim` :: `integer(1)`\cr
#'   The dimension of the embedding.
#' * `num_embeddings` :: `integer(1)`\cr
#'   The number of rows of the embedding table, i.e. the largest index that can be looked up.
#'   If this is not set, it is inferred from the task as the largest number of categories over its
#'   categorical features, which is what the categorical ingress produces codes for. It has to be
#'   set for inputs that do not come from a task's categorical features, such as a
#'   [`lazy_tensor`] of token ids.
#' * `padding_idx` :: `integer(1)`\cr
#'   An index whose embedding is fixed to zero and does not receive gradients. Default is `NULL`.
#' * `max_norm` :: `numeric(1)`\cr
#'   If set, embeddings with a larger norm are renormalized to it. Default is `NULL`.
#' * `norm_type` :: `numeric(1)`\cr
#'   The exponent of the norm used for `max_norm`. Default is `2`.
#' * `scale_grad_by_freq` :: `logical(1)`\cr
#'   Whether to scale the gradients by the inverse frequency of the indices in the batch.
#'   Default is `FALSE`.
#' * `sparse` :: `logical(1)`\cr
#'   Whether the gradient of the embedding table is a sparse tensor. Default is `FALSE`.
#'
#' @templateVar id nn_embedding
#' @templateVar param_vals embedding_dim = 10
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchEmbedding = R6Class("PipeOpTorchEmbedding",
  inherit = PipeOpTorchEmbeddingBase,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_embedding", param_vals = list()) {
      param_set = ps(
        embedding_dim = p_int(lower = 1L, tags = c("train", "required")),
        num_embeddings = p_int(lower = 1L, tags = "train"),
        padding_idx = p_int(lower = 1L, default = NULL, special_vals = list(NULL), tags = "train"),
        max_norm = p_dbl(lower = 0, default = NULL, special_vals = list(NULL), tags = "train"),
        norm_type = p_dbl(lower = 0, default = 2, tags = "train"),
        scale_grad_by_freq = p_lgl(default = FALSE, tags = "train"),
        sparse = p_lgl(default = FALSE, tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_embedding
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      private$.assert_num_embeddings(param_vals, task)
      # every index is replaced by its row of the table, so the shape gains a last dimension
      list(as.integer(c(shapes_in[[1L]], param_vals[["embedding_dim"]])))
    }
  )
)

#' @title Embedding Bag
#' @inherit torch::nn_embedding_bag description
#' @section nn_module:
#' Calls [`torch::nn_embedding_bag()`] when trained.
#'
#' The input is a `(batch, n)` tensor of indices, each row of which is one bag, and the output is
#' the `(batch, embedding_dim)` reduction of the embedded indices of each bag.
#' The indices start at 1, as they do everywhere in `torch`'s R interface.
#'
#' @section Parameters:
#' * `embedding_dim` :: `integer(1)`\cr
#'   The dimension of the embedding.
#' * `num_embeddings` :: `integer(1)`\cr
#'   The number of rows of the embedding table, inferred from the task when it is not set, see
#'   [`nn("embedding")`][mlr_pipeops_nn_embedding].
#' * `mode` :: `character(1)`\cr
#'   How the embeddings of a bag are reduced, one of `"sum"`, `"mean"` or `"max"`.
#'   Default is `"mean"`.
#' * `max_norm` :: `numeric(1)`\cr
#'   If set, embeddings with a larger norm are renormalized to it. Default is `NULL`.
#' * `norm_type` :: `numeric(1)`\cr
#'   The exponent of the norm used for `max_norm`. Default is `2`.
#' * `scale_grad_by_freq` :: `logical(1)`\cr
#'   Whether to scale the gradients by the inverse frequency of the indices in the batch.
#'   Default is `FALSE`. This is not supported for `mode = "max"`.
#' * `sparse` :: `logical(1)`\cr
#'   Whether the gradient of the embedding table is a sparse tensor. Default is `FALSE`.
#'   This is not supported for `mode = "max"`.
#' * `padding_idx` :: `integer(1)`\cr
#'   An index whose embedding is fixed to zero and which is skipped by the reduction.
#'   Default is `NULL`.
#'
#' @templateVar id nn_embedding_bag
#' @templateVar param_vals embedding_dim = 10
#' @template pipeop_torch_channels_default
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchEmbeddingBag = R6Class("PipeOpTorchEmbeddingBag",
  inherit = PipeOpTorchEmbeddingBase,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_embedding_bag", param_vals = list()) {
      param_set = ps(
        embedding_dim = p_int(lower = 1L, tags = c("train", "required")),
        num_embeddings = p_int(lower = 1L, tags = "train"),
        mode = p_fct(default = "mean", levels = c("sum", "mean", "max"), tags = "train"),
        max_norm = p_dbl(lower = 0, default = NULL, special_vals = list(NULL), tags = "train"),
        norm_type = p_dbl(lower = 0, default = 2, tags = "train"),
        scale_grad_by_freq = p_lgl(default = FALSE, tags = "train"),
        sparse = p_lgl(default = FALSE, tags = "train"),
        padding_idx = p_int(lower = 1L, default = NULL, special_vals = list(NULL), tags = "train")
      )
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_embedding_bag
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      # each row is one bag, which the module reduces to a single embedding
      assert_ndim(shape, 2L, self$id)
      private$.assert_num_embeddings(param_vals, task)
      list(as.integer(c(shape[[1L]], param_vals[["embedding_dim"]])))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals = super$.shape_dependent_params(shapes_in, param_vals, task)
      # torch rejects these combinations only once the backward pass runs into them
      if (identical(param_vals$mode, "max") && (isTRUE(param_vals$scale_grad_by_freq) || isTRUE(param_vals$sparse))) { # nolint
        stopf("PipeOp '%s': 'mode' = \"max\" cannot be combined with 'scale_grad_by_freq' or 'sparse'.",
          self$id)
      }
      # `include_last_offset` only means something together with the `offsets` argument of the
      # forward method, which a network's tensors do not carry
      param_vals$include_last_offset = FALSE
      param_vals
    }
  )
)

#' @include aaa.R
register_po("nn_embedding", PipeOpTorchEmbedding)
register_po("nn_embedding_bag", PipeOpTorchEmbeddingBag)
