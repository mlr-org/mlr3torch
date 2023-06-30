LearnerTorchTest1 = R6Class("LearnerTorchTest1",
  inherit = LearnerTorch,
  public = list(
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      properties = switch(task_type,
        regr = c(),
        classif = c("multiclass", "twoclass")
      )
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".test1"),
        label = "Test1 Learner",
        feature_types = c("numeric", "integer"),
        param_set = ps(bias = p_lgl(default = FALSE, tags = "train")),
        properties = properties,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners.test1"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals, defaults) {
      nout = if (task$task_type == "classif") length(task$class_names) else 1
      nn_linear(length(task$feature_names), nout, bias = param_vals$bias %??% FALSE)
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


LearnerTorchImageTest = R6Class("LearnerTorchImageTest",
  inherit = LearnerTorchImage,
  public = list(
    initialize = function(task_type, loss = t_loss("cross_entropy"), optimizer = t_opt("adam"), callbacks = list()) {
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".image_test"),
        param_set = ps(bias = p_lgl(default = FALSE, tags = "train")),
        label = "Test Learner Image",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        packages = "R6", # Just to check whether is is correctly passed
        man = "mlr3torch::mlr_learners.test"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals, defaults) {
      d = prod(param_vals$height, param_vals$width, param_vals$channels)
      nout = if (task$task_type == "classif") length(task$class_names) else 1
      nn_sequential(
        nn_flatten(),
        nn_linear(d, nout, bias = param_vals$bias %??% self$param_set$default$bias)
      )
    }
  )
)
