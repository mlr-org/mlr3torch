#' @title Classification AlexNet Learner
#' @author Lukas Burk
#' @name mlr_learners_classif.torch.alexnet
#'
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

      # Set param values that differ from default in tabnet_fit
      ps$values = list(
        pretrained = TRUE
      )

      super$initialize(
        id = "classif.torch.alexnet",
        packages = c("torchvision", "torch"),
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
      val_idx <- sample(task$nrow, floor(task$nrow * pars$valid_split))
      train_idx <- setdiff(seq_len(task$nrow), val_idx)

      # Check if sample sizes would be smaller than batch size
      if (min(length(train_idx), length(val_idx)) < pars$batch_size) {
        stop("batch_size larger than sample size")
      }

      train_ds <- img_dataset(task$data(), row_ids = train_idx, transform = pars$img_transform_train)
      valid_ds <- img_dataset(task$data(), row_ids = val_idx, transform = pars$img_transform_predict)

      train_dl <- torch::dataloader(train_ds, batch_size = pars$batch_size, shuffle = TRUE, drop_last = pars$drop_last)
      valid_dl <- torch::dataloader(valid_ds, batch_size = pars$batch_size, shuffle = FALSE, drop_last = pars$drop_last)

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
        luz::set_hparams(num_classes = train_ds$num_classes, pretrained = pars$pretrained) %>%
        luz::fit(train_dl, epochs = pars$epochs, valid_data = valid_dl)

      model_fit

    },

    .predict = function(task) {
      # get parameters with tag "predict"
      pars <- self$param_set$get_values(tags = "predict")
      pars_control <- self$param_set$get_values(tags = "control")

      # Drop control param from training pars
      pars <- pars[!(names(pars) %in% names(pars_control))]

      # TODO: See if reusing batch_size here is problematic
      test_ds <- img_dataset(task$data(), transform = pars$img_transform_predict)
      test_dl <- torch::dataloader(test_ds, batch_size = pars$batch_size, shuffle = FALSE, drop_last = FALSE)

      # Note on prediction:
      # Needs move to CPU to convert to R-native data structure
      # Calls luz' predict method, requires luz to be loaded
      model_pred <- predict(self$model, test_dl, verbose = pars_control$verbose)

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
