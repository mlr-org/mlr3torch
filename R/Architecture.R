#' @description
#' @export

Architecture = R6Class("Architecture",
  public = list(
    layers = NULL,
    initialize = function() {
      self$layers = list()
    },
    add = function(builder, param_vals = NULL) {
      layer = list(
        builder = builder,
        param_vals = param_vals
      )
      self$layers[[length(self$layers) + 1]] = assert_layer(layer)
      invisible(self)
    },
    print = function() {
      catf("<Architecture>")
      for (layer in self$layers) {
        catf(" <%s: %s>", get_private(layer[["operator"]])$.operator,
          format_named_list(layer[["param_vals"]]))
      }
    },

    build = function(task) {
      network = reduce_architecture(self, task)
      return(network)
    }
  )
)

format_named_list = function(l) {
  .f = function(x, y) {
    paste(y, x, sep = " = ")
  }
  paste(imap(l, .f), collapse = ", ")
}

reduce_architecture = function(architecture, task, input = NULL) {
  if (is.null(input)) {
    data_loader = make_dataloader(task, 1, "cpu") # TODO: don't build full dataloader here
    input = data_loader$.iter()$.next()
  }
  init = list(
    tensors = input,
    layers = list()
  )
  f = function(lhs, rhs) {
    tensors = lhs[["tensors"]]
    layers = lhs[["layers"]]
    layer_desc = rhs
    # TODO: Maybe we can use get(mlr_torchops$items), which would mean we don't have to
    # initialize the elements (we can simply use the .build method as a class method).
    # ATTENTION: We would have to take care of those TorchOp that require arguments
    # to be initialized and whose .build methods depends on that (e.g. TorchOpParallel)
    # either they get their own subclass or we simply check here (e.g. length(formalArgs))
    builder = layer_desc[["builder"]]
    # builder = get(layer_desc[["operator"]], envir = .__bobs__.)
    layer = builder(tensors, layer_desc[["param_vals"]], task)
    xs = tensors[startsWith(names(tensors), "x")]
    if (inherits(layer, "nn_tokenizer")) {
      x_new = with_no_grad(layer(input_num = xs[["x_num"]], input_cat = xs[["x_cat"]]))
    } else if (length(xs) == 1L) {
      x_new = layer(xs[[1L]])
    } else {
      stop("Not yet implemented!")
    }
    layers = append(layers, layer)
    tensors_new = list(x = x_new)
    if ("y" %in% names(tensors)) {
      tensors_new[["y"]] = tensors[["y"]]
    }
    output = list(
      tensors = tensors_new,
      layers = layers
    )
    return(output)
  }
  reduction = Reduce(f, architecture$layers, init)
  layers = reduction[["layers"]]
  model_output = reduction[["tensors"]]
  model = invoke(nn_sequential, .args = layers)
  return(list(model = model, output = model_output))
}

assert_layer = function(layer) {
  assert_names(names(layer), subset.of = c("builder", "param_vals"), must.include = "builder")
  assert_list(layer, min.len = 1L, max.len = 2L)
}

assert_layers = function(layers) {
  assert(map(layers, assert_layer))
}
