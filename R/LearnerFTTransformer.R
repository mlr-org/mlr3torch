
#' @title FT-Transformer
#' @description
#' Feature-Tokenizer Transformer.
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

      check_input_map = crate(function(input_map, task) {
        if (is.null(input_map)) {
          return(TRUE)
        }
        if (!is.character(input_map) || length(input_map) > 2L || !all(names(input_map) %in% c("num.input", "categ.input"))) {
          return("Parameter `input_map` must be a named `character()` of length 2, with names `num.input` and `categ.input`.")
        }
        return(TRUE)
      })

      private$.param_set_base = ps(
        n_blocks = p_int(lower = 0, tags = c("train", "required")),
        d_token = p_int(lower = 1L, tags = c("train", "required")),
        init_token = p_fct(init = "uniform", levels = c("uniform", "normal"), tags = c("train", "required")),
        input_map = p_uty(tags = "train", custom_check = check_input_map),
        shapes = p_uty(tags = "train")
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
      )
    }
  ),
  private = list(
    .block = NULL,
    .ingress_tokens = function(task, param_vals) {
      if ("lazy_tensor" %in% task$feature_types) {
        if (!all(task$feature_types$type == "lazy_tensor")) {
          stopf("Learner '%s' received an input task '%s' that is mixing lazy_tensors with other feature types.", self$id, task$id) # nolint
        }
        if (task$n_features > 2L) {
          stopf("Learner '%s' received an input task '%s' that has more than two lazy tensors.", self$id, task$id) # nolint
        }
        if (is.null(param_vals$input_map)) {
          stopf("Learner '%s' received an input task '%s' with lazy tensors, but no parameter 'input_map' was specified.", self$id, task$id) # nolint
        }
        output = list()

        # @SEBI: Continue here
        shape_num = param_vals$shapes[[]]
        shape_categ = param_vals$shapes$categ.input

        if (!is.null(shape_num)) {
          output$num.input = ingress_ltnsr(feature_name = task$feature_names, shape = shape_num)
        }


        shape = param_vals$shapes
        set_names(list(
          ingress_ltnsr(feature_name = task$feature_names)
        ), param_vals$input_map)
      }
      # TODO: Here we also need to handle lazy tensors.
      # In this case, we somehow need to have a parameter that specifies which lazy tensor
      # is the categorical one and which the numeric one.
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
      path_num = if (n_num_features(task) > 0L) {
        po("select", id = "select_num", selector = selector_type(c("numeric", "integer"))) %>>%
          po("torch_ingress_num", id = "num") %>>%
          nn("tokenizer_num", param_vals = list(
            d_token = param_vals$d_token,
            bias = TRUE,
            initialization = param_vals$init_token
          ))
      }

      path_categ = if (n_categ_features(task) > 0L) {
        po("select", id = "select_categ", selector = selector_type(c("factor", "ordered", "logical"))) %>>%
          po("torch_ingress_categ", id = "categ") %>>%
          nn("tokenizer_categ", param_vals = list(
            d_token = param_vals$d_token,
            bias = TRUE,
            initialization = param_vals$init_token
          ))
      }

      browser()

      graph_tokenizer = gunion(list(path_num, path_categ)) %>>%
        nn("merge_cat", param_vals = list(dim = 2))

      blocks = map(seq_len(param_vals$n_blocks), function(i) {
        block = private$.block$clone(deep = TRUE)
        block$id = sprintf("block_%i", i)
        if (i == 1) {
          block$param_set$values$is_first_layer = TRUE
        } else {
          block$param_set$values$is_first_layer = FALSE
        }
        if (i == param_vals$n_blocks) {
          block$param_set$values$query_idx = 1L
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

      model_descriptor_to_module(graph$train(task)[[1L]])
    }
  )
)

register_learner("regr.ft_transformer", LearnerTorchFTTransformer)
register_learner("classif.ft_transformer", LearnerTorchFTTransformer)
