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
  inherit = LearnerClassif,

  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps <- ParamSet$new(list(
        ParamLgl$new("pretrained",  default = TRUE, tags = "train"),
        ParamInt$new("num_threads", default = 1L, lower = 1L, upper = Inf, tags =  c("train", "control")),
        ParamInt$new("batch_size",  default = 256L, lower = 1L, upper = Inf, tags = "train"),
        ParamFct$new("loss",        default = "cross_entropy", levels = c("cross_entropy"), tags = "train"),
        ParamInt$new("epochs",      default = 5L,  lower = 1L, upper = Inf, tags = "train"),
        ParamLgl$new("drop_last",   default = TRUE, tags = "train"),
        ParamDbl$new("valid_split", default = 0.2, lower = 0, upper = 1, tags = "train"),
        #ParamDbl$new("learn_rate",  default = 0.02, lower = 0, upper = 1, tags = "train"),
        ParamInt$new("step_size", default = 1, lower = 1, upper = Inf, tags = "train"),
        # FIXME: Shoddy placeholder, needs more thinking
        ParamUty$new("img_transforms",  default = NULL, tags = "train"),
        # FIXME: Currently either 'adam' or arbitrary optimizer function according to docs
        ParamUty$new("optimizer",   default = "adam", tags = "train"),
        ParamLgl$new("verbose",     default = TRUE, tags = "control"),
        ParamFct$new("device",      default = "auto", levels = c("auto", "cpu", "cuda"), tags = "control")
      ))

      # Set param values that differ from default in tabnet_fit
      ps$values = list(
        pretrained = TRUE,
        num_threads = 1L,
        batch_size = 128L,
        loss = "cross_entropy",
        epochs = 10L,
        drop_last = TRUE,
        valid_split = 0.2,
        step_size = 1,
        # FIXME: Figure out transform placement
        img_transforms = NULL,
        optimizer = "adam",
        verbose = TRUE,
        device = "auto"
      )

      super$initialize(
        id = "classif.torch.alexnet",
        packages = c("torchvision", "torch"),
        feature_types = c("imageuri"),
        # FIXME: prob prediction missing
        predict_types = c("response"), #, "prob"),
        param_set = ps,
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

      train_ds <- img_dataset(task$data(), row_ids = train_idx, transform = pars$img_transforms)
      valid_ds <- img_dataset(task$data(), row_ids = val_idx, transform = pars$img_transforms)

      train_dl <- torch::dataloader(train_ds, batch_size = pars$batch_size, shuffle = TRUE, drop_last = pars$drop_last)
      valid_dl <- torch::dataloader(valid_ds, batch_size = pars$batch_size, shuffle = FALSE, drop_last = pars$drop_last)

      train_alexnet(
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

    },

    .predict = function(task) {
      # get parameters with tag "predict"
      pars = self$param_set$get_values(tags = "predict")
      pars_control = self$param_set$get_values(tags = "control")

      # Drop control par from training pars
      pars <- pars[!(names(pars) %in% names(pars_control))]

      # FIXME: hardcoded transform for prediction
      img_transforms <- function(img) {
        img %>%
          # first convert image to tensor
          torchvision::transform_to_tensor() %>%
          # # then move to the GPU (if available)
          (function(x) x$to(device = pars_control$device)) %>%
          # Required resize for alexnet
          torchvision::transform_resize(c(64,64))
      }

      # FIXME: Ad hoc dataloader from input task with 1 possibly huge batch
      test_ds <- img_dataset(task$data(), transform = img_transforms)
      test_dl <- torch::dataloader(test_ds, batch_size = 1, shuffle = TRUE, drop_last = FALSE)

      # Not sure if eval mode needed here
      self$model$eval()

      # FIXME: Collect class predictions / class probs
      pred_class <- integer(0)

      torch::with_no_grad({
        coro::loop(for (b in test_dl) {
          pred <- self$model(b[[1]]$to(device = pars_control$device))

          pred <- as.integer(pred$argmax(dim = 2))
          pred_class <- c(pred_class, pred)
        })
      })


      if (self$predict_type == "response") {
        targets <- task$data(cols = "target")[[1]]
        list(response = levels(targets)[pred_class])
      } else {
        #   pred = mlr3misc::invoke(predict, self$model, new_data = newdata,
        #                           type = "prob", .args = pars)
        #
        #   # Result will be a df with one column per variable with names '.pred_<level>'
        #   # we want the names without ".pred"
        #   names(pred) <- sub(pattern = ".pred_", replacement = "", names(pred))
        #
        #   list(prob = as.matrix(pred))
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

  if (device == "auto") device <- if (torch::cuda_is_available()) "cuda" else "cpu"

  model <- torchvision::model_alexnet(pretrained = pretrained, num_classes = num_classes)
  # model$to(device = device)

  # FIXME: Set lr, or maybe default is okay when using scheduler anyway
  optimizer <- switch(optimizer,
    "adam" = torch::optim_adam(model$parameters)
  )

  # FIXME: Parameterize gamma? Value copied verbatim from example docs
  scheduler <- torch::lr_step(optimizer, step_size = step_size, gamma = 0.95)

  loss_fn <- switch (loss,
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
    # Example code had $add(1) after topk presumably b/c 0-indexed in old versions?
    # k = 5 from example code
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

    # FIXME: Probably should pre-allocate loss / acc vectors at some point
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
    if (verbose) mlr3misc::catf("[epoch %d]: Loss = %3.2f, Acc= %3.2f \n", epoch, mean(l), mean(acc))
  }

  # Return model object after training loop
  return(model)
}
