#' @title Entrypoint to Torch Network
#'
#' @description
#' Use this as entry-point to mlr3torch-networks.
#'
#' @examples
#' @export
PipeOpTorchIngress = R6Class("PipeOpTorchIngress",
  inherit = PipeOp,
  public = list(
    #' @field feature_types (`character()`)\cr
    #'   The feature types used by this ingress operator.
    feature_types = NULL,
    initialize = function(id, param_set = ps(), param_vals = list(),
      input = data.table(name = "input", train = "Task", predict = "Task"),
      output = data.table(name = "output", train = "ModelDescriptor", predict = "Task"),
      packages = character(0), feature_types) {
      self$feature_types = feature_types
      lockBinding("feature_types", self)

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output,
        packages = packages
      )
     }
  ),
  private = list(
    .get_batchgetter = function(task, param_vals) stop("private$.get_batchgetter() must be implemented."),
    .shape = function(task, param_vals) stop("private$.shape() must be implemented."),
    .train = function(inputs) {
      task = inputs[[1]]
      param_vals = self$param_set$get_values()
      graph = as_graph(po("nop", id = self$id))
      batchgetter = private$.get_batchgetter(task, param_vals)


      ## In case the user is tempted to do things that will break in bad ways...
      ## But this test could also be left out.
      forbidden_env = parent.env(environment())
      test_env = environment(batchgetter)
      while (!isNamespace(test_env) && !identical(test_env, .GlobalEnv) && !identical(test_env, emptyenv())) {
        if (identical(test_env, forbidden_env)) {
          stopf(".get_batchgetter() must not return a member of the PipeOpTorchIngress object itself.")
        }
        test_env = parent.env(test_env)
      }

      if (!sum(task$feature_types$type %in% self$feature_types)) {
        stopf("%s with id '%s' got task with no feature types that it can process.", class(self)[[1L]], self$id)
      }


      ingress = TorchIngressToken(
        features = task$feature_names,
        batchgetter = batchgetter,
        shape = private$.shape(task, param_vals)
      )

      self$state = list(ingress$shape)  # PipeOp API requires us to only set this to some list.
      list(ModelDescriptor(
        graph = graph,
        ingress = structure(list(ingress), names = graph$input$name),
        task = task,
        .pointer = as.character(graph$output[, c("op.id", "channel.name"), with = FALSE]),
        .pointer_shape = ingress$shape
      ))
    },
    .predict = function(inputs) inputs
  )
)

#' @title Torch Ingress Token
#'
#' @param features (`character`)\cr
#'   Features on which the batchgetter will operate.
#' @param batchgetter (`function`)\cr
#'   Function with two arguments: `data` and `device`. This function is given
#'   the output of `Task$data(rows = batch_indices, cols = features)`
#'   and it should produce a tensor of shape `shape_out`.
#' @param shape (`integer`)\cr
#'   Shape that `batchgetter` will produce. Batch-dimension should
#'   be included as `NA`.
#' @return `TorchIngressToken` object.
#' @export
TorchIngressToken = function(features, batchgetter, shape) {
  assert_character(features, any.missing = FALSE)
  assert_function(batchgetter, args = c("data", "device"))
  assert_integerish(shape)
  structure(list(
    features = features,
    batchgetter = batchgetter,
    shape = shape
  ), class = "TorchIngressToken")
}

#' @export
print.TorchIngressToken = function(x, ...) {
  cat(sprintf("Ingress: Task[%s] --> Tensor(%s)\n", str_collapse(x$features, n = 3, sep = ","), str_collapse(x$shape)))
}


#' @title Torch Entry Point for Numeric Features
#' @description
#'
PipeOpTorchIngressNumeric = R6Class("PipeOpTorchIngressNumeric",
  inherit = PipeOpTorchIngress,
  public = list(
    feature_types = c("numeric", "integer"),
    initialize = function(id = "torch_ingress_num", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, feature_types = c("numeric", "integer"))
    }
  ),
  private = list(
    .shape = function(task, param_vals) {
      c(NA, sum(task$feature_types$type %in% self$feature_types))
    },
    .get_batchgetter = function(task, param_vals) {
      batchgetter_num
    }
  )
)

#' @title Batchgetter for Numeric Data
#'
#' @param data (`data.table`)\cr
#'   `data.table` to be converted to a `tensor`.
#' @export
batchgetter_num = function(data, device) {
  torch_tensor(
    data = as.matrix(data),
    dtype = torch_float(),
    device = device
  )
}

#' @include zzz.R
register_po("torch_ingress_num", PipeOpTorchIngressNumeric)

#' @title Torch Entry Point for Categorical Features
PipeOpTorchIngressCategorical = R6Class("PipeOpTorchIngressCategorical",
  inherit = PipeOpTorchIngress,
  public = list(
    initialize = function(id = "torch_ingress_cat", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, feature_types = c("factor", "ordered"))
    },
    speak = function() cat("I am the ingress cat, meow! ^._.^\n")
  ),
  private = list(
    .shape = function(task, param_vals) {
      c(NA, sum(task$feature_types$type %in% self$feature_types))
    },
    .get_batchgetter = function(task, param_vals) {
      batchgetter_categ
    }
  )
)

#' @title Batchgetter for categorical data
#'
#' @param data (`data.table`)\cr
#'   `data.table` to be converted to a `tensor`.
#' @export
batchgetter_categ = function(data, device) {
  torch_tensor(
    data = as.matrix(data[, lapply(.SD, as.integer)]),
    dtype = torch_long(),
    device = device
  )
}

register_po("torch_ingress_cat", PipeOpTorchIngressCategorical)

# @title Torch Entry Point for Images
# @description
# uses task with "imageuri" column and loads this as images.
# doesn't do any preprocessing or so (image resizing) and instead just errors if images don't fit.
# also no data augmentation etc.
PipeOpTorchIngressImages = R6Class("PipeOpTorchIngressImages",
  inherit = PipeOpTorchIngress,
  public = list(
    initialize = function(id = "torch_ingress_img", param_vals = list()) {
      param_set = ps(
        channels = p_int(1),
        pixels_height = p_int(1),
        pixels_width = p_int(1),
      )
      super$initialize(id = id, param_vals = param_vals)
    }
  ),
  private = list(
    .shape = function(task, param_vals) c(NA, param_vals$channels, param_vals$pixels_height, param_vals$pixels_width),
    .get_batchgetter = function(task, param_vals) {
      imgshape = c(param_vals$channels, param_vals$pixels_height, param_vals$pixels_width)
      crate(function(data, device) {
        tensors = lapply(data[[1]], function(uri) {
          tnsr = torchvision::transform_to_tensor(magick::image_read(uri))
          assert_true(identical(tnsr$shape, imgshape))
          torch_reshape(tnsr, imgshape)
        })
        torch_cat(tensors, dim = 1)$to(device = device)
      }, imgshape, .parent = topenv())
    }
  )
)
register_po("torch_ingress_img", PipeOpTorchIngressImages)
