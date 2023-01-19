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
#'
#' @section Parameters:
#'
#' @section Fields:
#'
#' @section Methods:
#'
#' @section Internals:
#'
#' @export
LearnerClassifTorchImage = R6Class("LearnerClassifTorchImage",
  inherit = LearnerClassifTorch,
  public = list(
    initialize = function(id, param_set, label, optimizer = t_opt("adam"), loss = t_loss("cross_entropy"),
      packages, module) {
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
        properties = c("weights", "twoclass", "multiclass", "hotstart"),
        label = label,
        optimizer = optimizer,
        loss = loss,
        packages = union(packages, "torchvision", "magick"),
        predict_types = c("response", "prob"),
        feature_types = "imageuri",
        man = "mlr3torch::mlr_learners_classif_image"
      )
    }
  ),
  private = list(
    .dataloader = function(task, param_vals) {
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

      ingress_token = TorchIngressToken(task$feature_names, batchgetter, imgshape)


      dataset = task_dataset(
        task,
        feature_ingress_tokens = list(ingress_token),
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device
      )
      dataloader(
        dataset = dataset,
        batch_size = param_vals$batch_size,
        drop_last = param_vals$drop_last,
        shuffle = param_vals$shuffle
      )
    }
  )
)
