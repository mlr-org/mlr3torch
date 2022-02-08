#' @title Classification AlexNet Learner
#' @author Lukas Burk
#' @name mlr_learners_classif.torch.alexnet
#'
#' @template class_learner
#' @templateVar id classif.torch.alexnet
#' @templateVar caller model_alexnet
#' @references
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
        # FIXME: Shoddy placeholder, needs more thinking
        # ParamUty$new("img_transform_train",  default = NULL, tags = "train"),
        # ParamUty$new("img_transform_val",  default = NULL, tags = "train"),
        # ParamUty$new("img_transform_predict",  default = NULL, tags = "predict")
      ))

      # Set param values that differ from default in tabnet_fit
      ps$values = list(
        pretrained = TRUE
        # FIXME: Figure out transform placement
        # img_transform_train = NULL,
        # img_transform_val = NULL,
        # img_transform_predict = NULL
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
      valid_ds <- img_dataset(task$data(), row_ids = val_idx, transform = pars$img_transform_val)

      train_dl <- torch::dataloader(train_ds, batch_size = pars$batch_size, shuffle = TRUE, drop_last = pars$drop_last)
      valid_dl <- torch::dataloader(valid_ds, batch_size = pars$batch_size, shuffle = FALSE, drop_last = pars$drop_last)

      ret <- train_alexnet(
        pretrained = pars$pretrained,
        train_dl = train_dl, valid_dl = valid_dl,
        num_classes = length(img_task$class_names),
        optimizer = pars$optimizer,
        #scheduler,
        step_size = pars$step_size,
        loss = pars$loss,
        epochs = pars$epochs,
        verbose = pars_control$verbose,
        device = pars_control$device
      )

      # Return model only, not sure where/how/if to store history
      # ret$history
      ret$model

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

#' Wrapper to fit alexnet within learner
#' @keywords internal
#' @return Trained nn_module object
train_alexnet <- function(
  pretrained = FALSE,
  train_dl, valid_dl,
  num_classes,
  optimizer = "adam",
  #scheduler,
  step_size = 1,
  loss,
  epochs,
  verbose,
  device
  ) {

  if (pretrained) {
    model <- torchvision::model_alexnet(pretrained = TRUE)
    model <- reset_last_layer(model, num_classes)
  } else {
    model <- torchvision::model_alexnet(pretrained = FALSE, num_classes = num_classes)
  }
  model$to(device = device)

  # FIXME: Set lr, or maybe default is okay when using scheduler anyway
  optimizer <- switch(optimizer,
    "adam" = torch::optim_adam(model$parameters)
  )

  # FIXME: Parameterize gamma? Value copied verbatim from example docs
  scheduler <- torch::lr_step(optimizer, step_size = step_size, gamma = 0.95)

  loss_fn <- switch(loss,
    "cross_entropy" = torch::nn_cross_entropy_loss()
  )

  # Training loop -----------------------------------------------------------

  train_step <- function(batch) {
    optimizer$zero_grad()
    output <- model(batch[[1]]$to(device = device))
    loss <- loss_fn(output, batch[[2]]$to(device = device))
    loss$backward()
    optimizer$step()
    loss
  }

  valid_step <- function(batch) {
    model$eval()
    pred <- model(batch[[1]]$to(device = device))
    # TODO: k = 5 from example code, parameterize?
    pred <- torch::torch_topk(pred, k = 5, dim = 2, TRUE, TRUE)[[2]]
    pred <- pred$to(device = torch::torch_device("cpu"))
    correct <- batch[[2]]$view(c(-1, 1))$eq(pred)$any(dim = 2)
    model$train()
    correct$to(dtype = torch::torch_float32())$mean()$item()
  }

  for (epoch in seq_len(epochs)) {

    if (verbose) {
      pb <- progress::progress_bar$new(
        total = length(train_dl),
        format = "[:bar] :eta Loss: :loss"
      )
    }

    # Note: Probably should pre-allocate loss / acc vectors at some point
    # Not trivial though, pre-allocating with number of batches would work
    # but indexing within coro::loop not possible since `b` does not
    # contain batch index
    l <- c()
    coro::loop(for (b in train_dl) {
      loss <- train_step(b)
      l <- c(l, loss$item())
      if (verbose) pb$tick(tokens = list(loss = mean(l)))
    })

    acc <- c()
    torch::with_no_grad({
      coro::loop(for (b in valid_dl) {
        accuracy <- valid_step(b)
        acc <- c(acc, accuracy)
      })
    })

    scheduler$step()
    if (verbose) cat(sprintf("[epoch %d]: Loss = %3f, Acc = %3f \n", epoch, mean(l), mean(acc)))

  }

  # Return model object after training loop, including history
  list(model = model, history = list(loss = l, acc = acc))
}
