#' @description
#' @export

Architecture = R6Class("Architecture",
  public = list(
    layers = NULL,
    initialize = function() {
      self$layers = list()
    },
    add = function(operator, param_vals = NULL) {
      layer = list(
        operator = operator,
        param_vals = param_vals
      )
      self$layers[[length(self$layers) + 1]] = assert_layer(layer)
      invisible(self)
    },
    print = function() {
      catf("<Architecture>")
      for (layer in self$layers) {
        catf(" <%s: %s>", layer[["operator"]], format_named_list(layer[["param_vals"]]))
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

reduce_architecture = function(architecture, task) {
  data_loader = make_dataloader(task, 1, "cpu") # TODO: don't build full dataloader here
  init = list(
    tensors = data_loader$.iter()$.next(),
    layers = list()
  )
  f = function(lhs, rhs) {
    tensors = lhs[["tensors"]]
    layers = lhs[["layers"]]
    layer_desc = rhs
    builder = mlr_torchops$get(layer_desc[["operator"]])$build
    # builder = get(layer_desc[["operator"]], envir = .__bobs__.)
    layer = builder(tensors, layer_desc[["param_vals"]], task)
    xs = tensors[startsWith(names(tensors), "x")]
    if (length(xs) == 1) { # almost always except for the encoder (maybe others later)
      xs = xs[[1]]
    }
    x_new = with_no_grad(layer(xs))
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
  assert_names(names(layer), subset.of = c("operator", "param_vals"), must.include = "operator")
  assert_list(layer, min.len = 1L, max.len = 2L)
}

assert_layers = function(layers) {
  assert(map(layers, assert_layer))
}
