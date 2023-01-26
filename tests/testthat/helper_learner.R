LearnerClassifTest1 = R6Class("LearnerClassifTest1",
  inherit = LearnerClassifTorchAbstract,
  public = list(
    initialize = function(optimizer = t_opt("adagrad"), loss = t_loss("cross_entropy")) {
      super$initialize(
        id = "classif.test1",
        label = "Test1 Classifier",
        feature_types = c("numeric", "integer"),
        param_set = ps(bias = p_lgl(default = FALSE, tags = "train")),
        properties = c("hotstart_forward", "multiclass", "twoclass"),
        predict_types = "response",
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners_classif.test1"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals, defaults) {
      nn_linear(length(task$feature_names), length(task$class_names), bias = param_vals$bias %??% FALSE)
    },
    .dataloader = function(task, param_vals, defaults) {
      ingress_token = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, length(task$feature_names)))
      dataset = task_dataset(
        task,
        feature_ingress_tokens = list(num = ingress_token),
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device %??% self$param_set$default$device
      )
      dl = dataloader(
        dataset = dataset,
        batch_size = param_vals$batch_size,
        drop_last = param_vals$drop_last %??% self$param_set$default$drop_last,
        shuffle = param_vals$shuffle %??% self$param_set$default$shuffle
      )
      return(dl)

    }
  )
)



LearnerClassifTorchImageTest = R6Class("LearnerClassifTorchImageTest",
  inherit = LearnerClassifTorchImage,
  public = list(
    initialize = function(loss = t_loss("cross_entropy"), optimizer = t_opt("adam")) {
      super$initialize(
        id = "classif.test",
        param_set = ps(bias = p_lgl(default = FALSE, tags = "train")),
        label = "Test Learner Image",
        packages = "R6" # Just to check whether is is correctly passed
      )
    }
  ),
  private = list(
    .network = function(task, param_vals, defaults) {
      d = prod(param_vals$height, param_vals$width, param_vals$channels)
      nn_sequential(
        nn_flatten(),
        nn_linear(d, length(task$class_names), bias = param_vals$bias %??% self$param_set$default$bias)
      )
    }
  )
)
