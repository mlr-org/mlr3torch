#' @title Torch Learner for classification
#' @author Lukas Burk
#' @name mlr_learners_classif.torch
#'
#' @usage NULL
#'
#' @template seealso_learner
#' @templateVar learner_name classif.torch
# @export
#' @examples
#' \dontrun{
#' library(mlr3)
#' library(mlr3torch)
#' }
LearnerClassifTorch = R6::R6Class("LearnerClassifTorch",
  inherit = LearnerClassif,

  public = list(
    initialize = function(
      id = "classif.torch",
      predict_types = c("response", "prob"),
      feature_types = c("integer", "numeric"),
      properties = c("twoclass", "multiclass"),
      packages = c("torch"),
      man = "mlr3torch::mlr_learners_classif.torch",
      param_set = ps()
    ) {
      ps <- ParamSet$new(list(
        ParamInt$new("num_threads", default = 1L, lower = 1L, upper = Inf, tags =  c("train", "control")),
        ParamInt$new("batch_size",  default = 128L, lower = 1L, upper = Inf, tags = c("train", "predict")),
        ParamFct$new("loss",        default = "cross_entropy", levels = torch_reflections$loss$classif, tags = "train"),
        ParamInt$new("epochs",      default = 5L,  lower = 1L, upper = Inf, tags = "train"),
        ParamLgl$new("drop_last",   default = TRUE, tags = "train"),
        ParamDbl$new("valid_split", default = 0.2, lower = 0, upper = 1, tags = "train"),
        ParamDbl$new("learn_rate",  default = 0.02, lower = 0, upper = 1, tags = "train"),
        #ParamInt$new("step_size", default = 1, lower = 1, upper = Inf, tags = "train"),
        ParamUty$new("img_transform_train",  default = NULL, tags = "train"),
        ParamUty$new("img_transform_predict",  default = NULL, tags = c("train", "predict")),
        ParamFct$new("optimizer",   default = "adam", levels = torch_reflections$optimizer, tags = "train"),
        ParamLgl$new("verbose",     default = TRUE, tags = "control"),
        ParamUty$new("device",      default = "cpu", custom_check = function(x) x %in% get_available_device(), tags = "control")
      ))

      # Set param values that differ from default in tabnet_fit
      ps$values = list(
        num_threads = 1L,
        batch_size = 128L,
        loss = "cross_entropy",
        epochs = 10L,
        drop_last = TRUE,
        valid_split = 0.2,
        learn_rate = 0.02,
        #step_size = 1,
        # FIXME: Figure out transform placement
        img_transform_train = NULL,
        img_transform_predict = NULL,
        optimizer = "adam",
        verbose = TRUE,
        device = "cpu"
      )

      super$initialize(
        id = assert_character(id, len = 1),
        param_set = ParamSetCollection$new(list(ps, param_set)),
        predict_types = assert_character(predict_types),
        feature_types = assert_character(feature_types),
        properties = assert_character(properties),
        packages = assert_character(packages),
        man = assert_character(man)
      )
    }
  ),

  private = list(

    .train = function(task) {
      # # get parameters for training
      # pars = self$param_set$get_values(tags = "train")
      # pars_control = self$param_set$get_values(tags = "control")
      #
      # # Drop control par from training pars
      # pars <- pars[!(names(pars) %in% names(pars_control))]
      #
      # # Set number of threads
      # torch::torch_set_num_threads(pars_control$num_threads)


    },

    .predict = function(task) {
      # get parameters with tag "predict"
      pars <- self$param_set$get_values(tags = "predict")
      pars_control <- self$param_set$get_values(tags = "control")

      # Drop control param from training pars
      pars <- pars[!(names(pars) %in% names(pars_control))]

      # FIXME: Ad hoc dataloader from input task with 1 possibly huge batch
      test_ds <- img_dataset(task$data(), transform = pars$img_transform_predict)
      test_dl <- torch::dataloader(test_ds, batch_size = task$nrow, shuffle = FALSE, drop_last = FALSE)

      # Not sure if eval mode needed here
      self$model$eval()

      # FIXME: Collect class predictions / class probs
      # Note on prediction:
      # batch_size should be higher probably, but i.e. single-sample predictions
      # should also be allowed. Maybe not even necessary to use a dl here?
      # devices: model should be called on batch with both on same device
      # but to be coerced to R-native data structure, tensors have to be
      # moved to cpu first

      if (self$predict_type == "response") {
        pred_class <- integer(0)

        torch::with_no_grad({
          coro::loop(for (b in test_dl) {
            pred <- self$model(b[[1]]$to(device = pars_control$device))

            # as.integer coercion requires tensor to be on CPU
            pred <- as.integer(pred$argmax(dim = 2)$to(device = "cpu"))
            pred_class <- c(pred_class, pred)
          })
        })
        targets <- task$data(cols = "target")[[1]]
        list(response = levels(targets)[pred_class])
      } else {
        # Initialize 0-row matrix with proper type + column num
        # colnames need to correspond to class names
        pred_prob <- matrix(
          NA_real_,
          ncol = length(task$class_names),
          nrow = 0,
          dimnames = list(NULL, task$class_names)
        )

        torch::with_no_grad({
          coro::loop(for (b in test_dl) {
            pred <- self$model(b[[1]]$to(device = pars_control$device))

            pred <- pred$softmax(dim = 2)$to(device = "cpu")
            pred <- as.matrix(pred)
            pred_prob <- rbind(pred_prob, pred)
          })
        })

        list(prob = pred_prob)
      }

    }
  )
)
