#' @title Entrypoint to Torch Network
#'
#' @usage NULL
#' @name mlr_pipeops_torch_ingress
#' @format [`R6Class`] inheriting from [`PipeOp`].
#'
#' @description
#' Use this as entry-point to mlr3torch-networks.
#' Unless you are an advanced user, you should not need to use this directly but [`PipeOpTorchIngressNumeric`],
#' [`PipeOpTorchIngressCategorical`] or [`PipeOpTorchIngressImage`].
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchIngress)`
#'
#' * `r roxy_param_id()`
#' * `r roxy_param_param_set()`
#' * `r roxy_param_param_vals()`
#' * `input` :: `data.table()`\cr
#'   The input channels for this `PipeOp`. See [`PipeOp`] for an explanation.
#' * `output` :: `data.table()`\cr
#'   The output channels for this `PipeOp`. See [`PipeOp`] for an explanation.
#' * `packages` :: `character() cr
#'   The packages this `PipeOpTorchIngress` depends on.
#'
#' @section Input and Output Channels: `r roxy_pipeop_torch_channels_default()`
#' @section State:
#' The state is set to the input shape.
#'
#' @section Parameters:
#' Defined by the construction argument `param_set`.
#'
#' @section Fields:
#' * `feature_types` :: `character(1)`\cr
#'   The features types that can be consumed by this `PipeOpTorchIngress`.
#' @section Methods: `r roxy_pipeop_torch_methods_default()`
#'
#' @section Internals:
#' Creates an object of class [`TorchIngressToken`] for the given task.
#' The purpuse of this is to store the information on how to construct the torch dataloader from the task for this
#' entry point of the network. This is done in the function [`task_dataset()`].
#'
#' @family PipeOp
#' @export
PipeOpTorchIngress = R6Class("PipeOpTorchIngress",
  inherit = PipeOp,
  public = list(
    feature_types = NULL,
    initialize = function(id, param_set = ps(), param_vals = list(), packages = character(0), feature_types) {
      self$feature_types = feature_types
      lockBinding("feature_types", self)
      ps_tmp = ps(
        select = p_lgl(default = FALSE, tags = "train")
      )

      if (param_set$length > 0) {
        param_set = ParamSetCollection$new(list(param_set, ps_tmp))
      } else {
        param_set = ps_tmp
      }
      param_set$values$select = FALSE

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = data.table(name = "input", train = "Task", predict = "Task"),
        output = data.table(name = "output", train = "ModelDescriptor", predict = "Task"),
        packages = packages
      )
     }
  ),
  private = list(
    .get_batchgetter = function(task, param_vals) stop("private$.get_batchgetter() must be implemented."),
    .shape = function(task, param_vals) stop("private$.shape() must be implemented."),
    .train = function(inputs) {
      pv = self$param_set$get_values(tags = "train")
      if (any(task$missings())) {
        # NAs are converted to their underlying machine representation when calling `torch_tensor()`
        # https://github.com/mlverse/torch/issues/933
        stopf("No missing values allowed in task '%s'.", task$id)
      }
      task = inputs[[1]]
      param_vals = self$param_set$get_values()
      graph = as_graph(po("nop", id = self$id))
      batchgetter = private$.get_batchgetter(task, param_vals)

      # In case the user is tempted to do things that will break in bad ways...
      # But this test could also be left out.
      forbidden_env = parent.env(environment())
      test_env = environment(batchgetter)
      while (!isNamespace(test_env) && !identical(test_env, .GlobalEnv) && !identical(test_env, emptyenv())) {
        if (identical(test_env, forbidden_env)) {
          stopf(".get_batchgetter() must not return a member of the PipeOpTorchIngress object itself.")
        }
        test_env = parent.env(test_env)
      }

      if (pv$select) {
        task = po("select", selector = selector_type(self$feature_types))$train(list(task))[[1L]]
      } else {
        if (!all(task$feature_types$type %in% self$feature_types)) {
          stopf("Task contains features of type %s, but only %s are allowed. Use parameter `select` to avoid this.",
            paste0(unique(task$feature_types$type[!(task$feature_types$type %in% self$feature_types)]), collapse = ", "),
            paste0(self$feature_types, collapse = ", ")
          )
        }
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
#' @description
#' This function creates an S3 class of class `"TorchIngressToken"`, which is an internal data structure.
#' It contains the (meta-)information of how a batch is generated from a [`Task`] and fed into an entry point
#' of the neural network. It is stored as the `ingress` field in a [`ModelDescriptor`].
#'
#' @param features (`character`)\cr
#'   Features on which the batchgetter will operate.
#' @param batchgetter (`function`)\cr
#'   Function with two arguments: `data` and `device`. This function is given
#'   the output of `Task$data(rows = batch_indices, cols = features)`
#'   and it should produce a tensor of shape `shape_out`.
#' @param shape (`integer`)\cr
#'   Shape that `batchgetter` will produce. Batch-dimension should be included as `NA`.
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
#' @usage NULL
#' @name mlr_pipeops_torch_ingress_numeric
#' @format [`R6Class`] object inheriting from [`PipeOpTorchIngress`] / [`PipeOp`].
#'
#' @description
#' Ingress PipeOp that represents a numeric (`integer()` and `numeric()`) entry point to a torch network.
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchIngressNumeric)`
#'
#' * `r roxy_param_id()`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels:
#' `r roxy_pipeop_torch_channels_default()`
#' @inheritSection mlr_pipeops_torch_ingress State
#' @section Parameters:
#' * `select` :: `logical(1)`\cr
#'   Whether `PipeOp` should selected the supported feature types. Otherwise it will err, when receiving tasks
#'   with unsupported feature types.
#' @section Fields:
#' Only fields inherited from [`PipeOpTorchIngress`] / [`PipeOp`].
#' @section Methods:
#' Only methods inherited from [`PipeOpTorchIngress`] / [`PipeOp`].
#' @section Internals:
#' Uses [batchgetter_num()].
#'
#' @export
#' @family PipeOps
#' @examples
#' # We set select to TRUE because the data contains factors as well
#' po_ingress = po("torch_ingress_num", select = TRUE)
#' task = tsk("mtcars")
#' # The output is a TorchIngressToken
#' token = po_ingress$train(list(task))[[1L]]
#' ingress = token$ingress[[1L]]
#' ingress$batchgetter(task$data(1:5, ingress$features), "cpu")
PipeOpTorchIngressNumeric = R6Class("PipeOpTorchIngressNumeric",
  inherit = PipeOpTorchIngress,
  public = list(
    initialize = function(id = "torch_ingress_num", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, feature_types = c("numeric", "integer"))
    }
  ),
  private = list(
    .shape = function(task, param_vals) {
      # Note that this function can only be called successfully if either select is TRUE or the task contains
      # only integers and numerics. In both cases the formula below is correct
      c(NA, sum(task$feature_types$type %in% self$feature_types))
    },
    .get_batchgetter = function(task, param_vals) {
      batchgetter_num
    }
  )
)

#' @title Batchgetter for Numeric Data
#'
#' @description
#' Converts a data frame of numeric data into a float tensor.
#'
#' @param data (`data.table()`)\cr
#'   `data.table` to be converted to a `tensor`.
#' @param device (`character(1)`)\cr
#'   The device on which the tensor should be created.
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
#' @usage NULL
#' @name mlr_pipeops_torch_ingress_cat
#' @format [`R6Class`] inheriting from [`PipeOpTorchIngress`]/[`PipeOpTorch`].
#'
#' @description
#' Ingress PipeOp that represents a categorical (`factor()`, `ordered()` and `logical()`) entry point to a torch network.
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchIngressCategorical)`
#'
#' * `r roxy_param_id()`
#' * `r roxy_param_param_vals()`
#'
#' @section Input and Output Channels:
#' `r roxy_pipeop_torch_channels_default()`
#' @section State:
#' `r roxy_pipeop_torch_state_default()`
#' @section Parameters:
#' * `select` :: `logical(1)`\cr
#'   Whether `PipeOp` should selected the supported feature types. Otherwise it will err on receiving tasks
#'   with unsupported feature types.
#' @section Fields:
#' Only fields inherited from [`PipeOpTorchIngress`] / [`PipeOp`].
#' @section Methods:
#' Only methods inherited from [`PipeOpTorchIngress`] / [`PipeOp`].
#' @section Internals:
#' Uses [`batchgetter_categ()`].
#' @family PipeOps
#' @export
#' @examples
# We set select to TRUE because the data contains factors as well
#' po_ingress = po("torch_ingress_cat", select = TRUE)
#' task = tsk("german_credit")
#' # The output is a TorchIngressToken
#' token = po_ingress$train(list(task))[[1L]]
#' ingress = token$ingress[[1L]]
#' ingress$batchgetter(task$data(1, ingress$features), "cpu")
PipeOpTorchIngressCategorical = R6Class("PipeOpTorchIngressCategorical",
  inherit = PipeOpTorchIngress,
  public = list(
    initialize = function(id = "torch_ingress_cat", param_vals = list()) {
      super$initialize(id = id, param_vals = param_vals, feature_types = c("factor", "ordered", "logical"))
    },
    speak = function() cat("I am the ingress cat, meow! ^._.^\n")
  ),
  private = list(
    .shape = function(task, param_vals) {
      c(NA, sum(task$feature_types$type %in% self$feature_types))
    },
    .get_batchgetter = function(task, param_vals) {
      # Note that this function can only be called successfully if either select is TRUE or the task contains
      # only factors and logicals. In both cases the formula below is correct
      batchgetter_categ
    }
  )
)

#' @title Batchgetter for categorical data
#'
#' @description
#' Converts a data frame of categorical data into a long tensor.
#'
#' @param data (`data.table`)\cr
#'   `data.table` to be converted to a `tensor`.
#' @param device (`character(1)`)\cr
#'   The device.
#' @export
batchgetter_categ = function(data, device) {
  torch_tensor(
    data = as.matrix(data[, lapply(.SD, as.integer)]),
    dtype = torch_long(),
    device = device
  )
}

register_po("torch_ingress_cat", PipeOpTorchIngressCategorical)

#' @title Torch Entry Point for Images
#' @usage NULL
#' @name mlr_pipeops_torch_ingress_img
#' @format [`R6Class`] inheriting from [`PipeOpTorchIngress`] and [`PipeOp`].
#'
#' @description
#' uses task with "imageuri" column and loads this as images.
#' doesn't do any preprocessing or so (image resizing) and instead just errors if images don't fit.
#' also no data augmentation etc.
#'
#' @section Construction:
#' `r roxy_construction(PipeOpTorchIngressImage)`
#'
#' @section Input and Output Channels:
#' `r roxy_pipeop_torch_channels_default()`
#' @section State:
#' `r roxy_pipeop_torch_state_default()`
#' @section Parameters:
#' * `select` :: `logical(1)`\cr
#'   Whether `PipeOp` should selected the supported feature types. Otherwise it will err, when receiving tasks
#'   with unsupported feature types.
#' * `channels` :: `integer(1)`\cr
#'   The number of input channels.
#' * `height` :: `integer(1)`\cr
#'   The height of the pixels.
#' * `width` :: `integer(1)`\cr
#'   The width of the pixels.
#' @section Fields:
#' Only fields inherited from [`PipeOpTorchIngress`] / [`PipeOp`].
#' @section Methods:
#' Only methods inherited from [`PipeOpTorchIngress`] / [`PipeOp`].
#' @section Internals:
#' Uses [`magick::image_read()`]'to load the image.
#'
#' @family PipeOp
#'
#' @export
#' @examples
#' po_ingress = po("torch_ingress_img", channels = 3, height = 64, width = 64)
#' task = tsk("tiny_imagenet")
#' token = po_ingress$train(list(task))[[1L]]
#' ingress = token$ingress[[1L]]
#' ingress$batchgetter(task$data(1, ingress$features), "cpu")
PipeOpTorchIngressImage = R6Class("PipeOpTorchIngressImage",
  inherit = PipeOpTorchIngress,
  public = list(
    initialize = function(id = "torch_ingress_img", param_vals = list(), param_set = param_set) {
      param_set = ps(
        channels = p_int(1, tags = "required"),
        height = p_int(1, tags = "required"),
        width = p_int(1, tags = "required")
      )
      super$initialize(id = id, param_vals = param_vals, param_set = param_set, feature_types = "imageuri")
    }
  ),
  private = list(
    .shape = function(task, param_vals) c(NA, param_vals$channels, param_vals$height, param_vals$width),
    .get_batchgetter = function(task, param_vals) {
      imgshape = c(param_vals$channels, param_vals$height, param_vals$width)
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
register_po("torch_ingress_img", PipeOpTorchIngressImage)
