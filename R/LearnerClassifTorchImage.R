#' @title Image Classification Network
#'
#' @usage  NULL
#' @name mlr_learners_classif_torch_image
#' @format `r roxy_format(LearnerClassifTorchImage)`
#'
#' @description
#' Base Class for Image Classification Learners.
#'
#' @section Construction: `r roxy_construction(LearnerClassifTorchImage)`
#'
#' @section Inheriting:
#' To inherit from this class, one should overwrite the private `$.network()` to return a [`nn_module`] that has
#' one argument in its forward method.
#'
#' @section Parameters:
#' Parameters include those inherited from [`LearnerClassifTorchAbstract`], the `param_set` construction argument, as
#' well as:
#'
#' * `channels` :: `integer(1)` \cr
#'   The number of input channels.
#' * `height` :: `integer(1)` \cr
#'   The height of the input image.
#' * `width` :: `integer(1)` \cr
#'   The width of the input image.
#'
#' @section Fields:
#'
#' @section Methods:
#'
#' @section Internals:
#'
#' @export
LearnerClassifTorchImage = R6Class("LearnerClassifTorchImage",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(id, param_set, label, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      packages = c("torchvision", "magick"), man) {
      assert_param_set(param_set)
      predefined_set = ps(
        channels   = p_int(1, tags = c("train", "required")),
        height     = p_int(1, tags = c("train", "required")),
        width      = p_int(1, tags = c("train", "required"))
      )

      if (param_set$length) {
        param_set$add(predefined_set)
      } else {
        param_set = predefined_set
      }

      super$initialize(
        id = id,
        label = label,
        optimizer = optimizer,
        properties = c("twoclass", "multiclass", "hotstart_forward"),
        loss = loss,
        param_set = param_set,
        packages = packages,
        predict_types = c("response", "prob"),
        feature_types = "imageuri",
        man = "mlr3torch::mlr_learners_classif_image"
      )
    }
  ),
  private = list(
    .dataset = function(task, param_vals) {
      assert_true(length(task$feature_names) == 1)
      # TODO: Maybe we want to be more careful here to avoid changing parameters between train and predict
      imgshape = c(param_vals$channels, param_vals$height, param_vals$width)

      batchgetter = crate(function(data, device) {
        tensors = lapply(data[[1]], function(uri) {
          tnsr = torchvision::transform_to_tensor(magick::image_read(uri))
          assert_true(identical(tnsr$shape, imgshape))
          torch_reshape(tnsr, imgshape)$unsqueeze(1)
        })
        torch_cat(tensors, dim = 1)$to(device = device)
      }, imgshape, .parent = topenv())

      ingress_tokens = list(image = TorchIngressToken(task$feature_names, batchgetter, imgshape))

      task_dataset(
        task,
        feature_ingress_tokens = ingress_tokens,
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device
      )
    }
  )
)
