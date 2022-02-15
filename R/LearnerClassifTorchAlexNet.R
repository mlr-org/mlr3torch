#' @title Classification AlexNet Learner
#' @author Lukas Burk
#' @name mlr_learners_classif.torch.alexnet
#' @usage NULL
#' @template class_learner
#' @templateVar id classif.torch.alexnet
#' @templateVar caller model_alexnet
#' @references
#' Krizhevsky, A. One weird trick for parallelizing convolutional neural
#' networks. arXiv:1404.5997 (2014).
#'
#' @template seealso_learner
#' @export
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3torch)
#' }
LearnerClassifTorchAlexNet = R6::R6Class("LearnerClassifTorchAlexNet",
  inherit = LearnerClassifTorch,

  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ParamSet$new(list(
        ParamLgl$new("pretrained", default = TRUE, tags = "train")
      ))

      # Set param values to defaults above
      ps$values = list(
        pretrained = TRUE
      )

      super$initialize(
        id = "classif.torch.alexnet",
        packages = c("torchvision", "torch", "luz"),
        param_set = ps,
        feature_types = c("imageuri"),
        predict_types = c("response", "prob"),
        properties = c("multiclass", "twoclass"),
        man = "mlr3torch::mlr_learners_classif.torch.alexnet"
      )
    }
  ),

  private = list(

    .train = function(task) {
      # get parameters for training
      pars = self$param_set$get_values(tags = "train")
      pars_control = self$param_set$get_values(tags = "control")

      # Drop control par from training pars
      pars <- pars[!(names(pars) %in% names(pars_control))]

      # Set number of threads
      torch::torch_set_num_threads(pars_control$num_threads)

      # set column names to ensure consistency in fit and predict
      # (pasted from tabnet learner, unlikely to be useful?)
      # self$state$feature_names = task$feature_names

      # Validation split & datasets/loaders-------------------------------------
      dls <- make_dl_from_task(
        task,
        valid_split = pars$valid_split,
        batch_size = pars$batch_size,
        transform_train = pars$img_transform_train,
        transform_val = pars$img_transform_predict,
        drop_last = pars$drop_last
      )

      model <- torch::nn_module(
        initialize = function(num_classes, pretrained = TRUE) {

          if (pretrained) {
            self$model <- torchvision::model_alexnet(pretrained = TRUE)
            self$model <- reset_last_layer(self$model, num_classes)
          } else {
            self$model <- torchvision::model_alexnet(pretrained = FALSE, num_classes = num_classes)
          }

        },
        forward = function(x) {
          self$model$forward(x)
        }
      )

      model_fit <- luz::setup(
          module = model,
          loss = get_torch_loss(pars$loss),
          optimizer = get_torch_optimizer(pars$optimizer),
          metrics = list(
            luz::luz_metric_accuracy()
          )
        ) %>%
        # Pass arguments to the nn_module initialize method
        luz::set_hparams(num_classes = length(task$class_names), pretrained = pars$pretrained) %>%
        # Pass arguments to the optimizer
        luz::set_opt_hparams(lr = pars$learn_rate) %>%
        luz::fit(
          data = dls$train,
          epochs = pars$epochs,
          # accelerator = luz::accelerator(),
          # Only pass validation dl if it exists, hinging on valid_split
          valid_data = if (pars$valid_split > 0) dls$val else NULL
        )

      model_fit

    },

    .predict = function(task) {
      # get parameters with tag "predict"
      pars <- self$param_set$get_values(tags = "predict")
      pars_control <- self$param_set$get_values(tags = "control")

      # Drop control param from training pars
      pars <- pars[!(names(pars) %in% names(pars_control))]

      dls <- make_dl_from_task(
        task,
        # Set to 0 to create only one dl: dls$train
        valid_split = 0,
        batch_size = pars$batch_size,
        # Only 'train' dl will be created, but with predict transformations
        transform_train = pars$img_transform_predict,
        drop_last = FALSE
      )
      # Note on prediction:
      # Needs move to CPU to convert to R-native data structure
      # Calls luz' predict method, requires luz to be loaded
      model_pred <- predict(self$model, dls$train, verbose = pars_control$verbose)

      if (self$predict_type == "response") {

        pred_class <- as.integer(model_pred$argmax(dim = 2)$to(device = "cpu"))

        # FIXME: Store class labels in learner based on input task
        # This is a "make it kind of work"-level hack
        targets <- task$data(cols = "target")[[1]]

        list(response = levels(targets)[pred_class])
      } else {

        pred_matrix <- as.matrix(model_pred$softmax(dim = 2)$to(device = "cpu"))

        # FIXME: Same fragile class name hack
        colnames(pred_matrix) <- task$class_names

        list(prob = pred_matrix)
      }

    }
  )
)
