LearnerClassifTorch1 = R6Class("LearnerClassifTorch1",
  inherit = LearnerClassifTorch,
  public = list(
    initialize = function(optimizer = t_opt("adagrad"), loss = t_loss("cross_entropy"), 
      callbacks = list()) {
      param_set = ps(bias = p_lgl(tags = c("required", "train")))
      param_set$values = list(bias = FALSE)
      super$initialize(
        id = "classif.test1",
        label = "Torch1 Classifier",
        feature_types = c("numeric", "integer"),
        param_set = param_set,
        properties = c("multiclass", "twoclass"),
        predict_types = "response",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        man = "mlr3torch::mlr_learners_classif.test1"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals, defaults) {
      nn_linear(length(task$feature_names), length(task$class_names), bias = param_vals$bias)
    },
    .dataloader = function(task, param_vals, defaults) {
      ingress_token = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, length(task$feature_names)))
      dataset = task_dataset(
        task,
        feature_ingress_tokens = list(num = ingress_token),
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device
      )
      dl = dataloader(
        dataset = dataset,
        batch_size = param_vals$batch_size,
        shuffle = param_vals$shuffle
      )
      return(dl)

    }
  )
)

LearnerRegrTorch1 = R6Class("LearnerRegrTorch1",
  inherit = LearnerRegrTorch,
  public = list(
    initialize = function(optimizer = t_opt("adagrad"), loss = t_loss("mse")) {
      param_set = ps(bias = p_lgl(tags = c("required", "train")))
      param_set$values = list(bias = FALSE)
      super$initialize(
        id = "regr.test1",
        label = "Test1 Regressor",
        feature_types = c("numeric", "integer"),
        param_set = param_set,
        predict_types = "response",
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_regr.test1"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals, defaults) {
      nn_linear(length(task$feature_names), 1, bias = param_vals$bias)
    },
    .dataloader = function(task, param_vals, defaults) {
      ingress_token = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, length(task$feature_names)))
      dataset = task_dataset(
        task,
        feature_ingress_tokens = list(num = ingress_token),
        target_batchgetter = target_batchgetter("regr"),
        device = param_vals$device
      )
      dl = dataloader(
        dataset = dataset,
        batch_size = param_vals$batch_size,
        shuffle = param_vals$shuffle
      )
      return(dl)
    }
  )
)



LearnerClassifTorchImageTest = R6Class("LearnerClassifTorchImageTest",
  inherit = LearnerClassifTorchImage,
  public = list(
    initialize = function(loss = t_loss("cross_entropy"), optimizer = t_opt("adam"), callbacks = list()) {

      param_set = ps(bias = p_lgl(tags = c("train", "required")))
      param_set$values = list(bias = FALSE)
      super$initialize(
        id = "classif.image_test",
        param_set = param_set,
        label = "Test Learner Image",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        packages = "R6", # Just to check whether is is correctly passed
        man = "mlr3torch::mlr_learners_classif.test"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals, defaults) {
      d = prod(param_vals$height, param_vals$width, param_vals$channels)
      nn_sequential(
        nn_flatten(),
        nn_linear(d, length(task$class_names), bias = param_vals$bias)
      )
    }
  )
)
