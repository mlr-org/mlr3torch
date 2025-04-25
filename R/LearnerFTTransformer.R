
#' @title FT-Transformer
#' @description
#' Feature-Tokenizer Transformer for tabular data that can either work on [`lazy_tesor`] inputs
#' or on standard tabular features.
#' @references
#' `r format_bib("gorishniy2021revisiting")`
#' @export
LearnerTorchFTTransformer = R6Class("LearnerTorchFTTransformer",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      private$.block = PipeOpTorchFTTransformerBlock$new()

      check_ingress_tokens = crate(function(ingress_tokens, task) {
        if (is.null(ingress_tokens)) {
          return(TRUE)
        }
        msg = check_list(ingress_tokens, types = "TorchIngressToken", min.len = 1L, names = "unique")
        if (!isTRUE(msg)) {
          return(msg)
        }
        check_subset(names(ingress_tokens), c("num.input", "categ.input"))
      })

      private$.param_set_base = ps(
        n_blocks = p_int(lower = 0, tags = c("train", "required")),
        d_token = p_int(lower = 1L, tags = c("train", "required")),
        cardinalities = p_int(lower = 1L, tags = "train"),
        init_token = p_fct(init = "uniform", levels = c("uniform", "normal"), tags = c("train", "required")),
        ingress_tokens = p_uty(tags = "train", custom_check = check_ingress_tokens)
      )
      param_set = alist(private$.block$param_set, private$.param_set_base)

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".ft_transformer"),
        label = "FT-Transformer",
        param_set = param_set,
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.ft_transformer",
        feature_types = c("numeric", "integer", "logical", "factor", "ordered", "lazy_tensor"),
        # Because the CLS token does resizing that depends dynamically on the input shape,
        # specifically, the batch size
        jittable = FALSE
      )
    }
  ),
  private = list(
    .block = NULL,
    .ingress_tokens = function(task, param_vals) {
      if ("lazy_tensor" %in% task$feature_types$type) {
        if (!all(task$feature_types$type == "lazy_tensor")) {
          stopf("Learner '%s' received an input task '%s' that is mixing lazy_tensors with other feature types.", self$id, task$id) # nolint
        }
        if (task$n_features > 2L) {
          stopf("Learner '%s' received an input task '%s' that has more than two lazy tensors.", self$id, task$id) # nolint
        }
        if (is.null(param_vals$ingress_tokens)) {
          stopf("Learner '%s' received an input task '%s' with lazy tensors, but no parameter 'ingress_tokens' was specified.", self$id, task$id) # nolint
        }

        ingress_tokens = param_vals$ingress_tokens
        row = task$head(1L)
        for (i in seq_along(ingress_tokens)) {
          feat = ingress_tokens[[i]]$features(task)
          if (!length(feat) == 1L) {
            stopf("Learner '%s' received an input task '%s' with lazy tensors, but the ingress token '%s' does not select exactly one feature.", self$id, task$id, names(ingress_tokens)[[i]]) # nolint
          }
          if (is.null(ingress_tokens[[i]]$shape)) {
            ingress_tokens[[i]]$shape = lazy_shape(row[[feat]])
          }
          if (is.null(ingress_tokens[[i]]$shape)) {
            stopf("Learner '%s' received an input task '%s' with lazy tensors, but neither the ingress token for '%s', nor the 'lazy_tensor' specify the shape, which makes it impossible to build the network.", self$id, task$id, feat) # nolint
          }
        }
        return(ingress_tokens)
      }
      num_features = n_num_features(task)
      categ_features = n_categ_features(task)
      output = list()
      if (num_features > 0L) {
        output$num.input = ingress_num(shape = c(NA, num_features))
      }
      if (categ_features > 0L) {
        output$categ.input = ingress_categ(shape = c(NA, categ_features))
      }
      output
    },
    .network = function(task, param_vals) {
      its = private$.ingress_tokens(task, param_vals)
      mds = list()
      path_num = if (!is.null(its$num.input)) {
        mds$tokenizer_num.input = ModelDescriptor(
          po("nop", id = "num"),
          its["num.input"],
          task$clone(deep = TRUE)$select(its[["num.input"]]$feature(task)),
          pointer = c("num", "output"),
          pointer_shape = its[["num.input"]]$shape
        )
        nn("tokenizer_num",
          d_token = param_vals$d_token,
          bias = TRUE,
          initialization = param_vals$init_token
        )
      }
      path_categ = if (!is.null(its$categ.input)) {
        mds$tokenizer_categ.input = ModelDescriptor(
          po("nop", id = "categ"),
          its["categ.input"],
          task$clone(deep = TRUE)$select(its[["categ.input"]]$feature(task)),
          pointer = c("categ", "output"),
          pointer_shape = its[["categ.input"]]$shape
        )
        nn("tokenizer_categ",
          d_token = param_vals$d_token,
          bias = TRUE,
          initialization = param_vals$init_token,
          param_vals = discard(param_vals["cardinalities"], is.null)
        )
      }

      input_paths = discard(list(path_num, path_categ), is.null)

      graph_tokenizer = if (length(input_paths) == 1L) {
        input_paths[[1L]]
      } else {
        gunion(input_paths) %>>%
          nn("merge_cat", param_vals = list(dim = 2))
      }

      blocks = map(seq_len(param_vals$n_blocks), function(i) {
        block = private$.block$clone(deep = TRUE)
        block$id = sprintf("block_%i", i)
        if (i == 1) {
          block$param_set$values$is_first_layer = TRUE
        } else {
          block$param_set$values$is_first_layer = FALSE
        }
        if (i == param_vals$n_blocks) {
          block$param_set$values$query_idx = -1L
        }
        block
      })

      blocks = if (length(blocks) == 1L) {
        Reduce(`%>>%`, blocks)
      }

      graph = graph_tokenizer %>>%
        nn("ft_cls", initialization = "uniform") %>>%
        blocks %>>%
        nn("fn", fn = function(x) x[, -1]) %>>%
        nn("layer_norm", dims = 1) %>>%
        nn("relu") %>>%
        nn("head")

      model_descriptor_to_module(graph$train(mds, FALSE)[[1L]])
    }
  )
)

register_learner("regr.ft_transformer", LearnerTorchFTTransformer)
register_learner("classif.ft_transformer", LearnerTorchFTTransformer)
