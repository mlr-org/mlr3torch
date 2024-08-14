library(R6)

LearnerTorchTest1 = R6Class("LearnerTorchTest1",
  inherit = LearnerTorch,
  public = list(
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      properties = switch(task_type,
        regr = c(),
        classif = c("multiclass", "twoclass")
      )
      param_set = ps(bias = p_lgl(tags = c("required", "train")))
      param_set$values = list(bias = FALSE)
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".test1"),
        label = "Test1 Learner",
        feature_types = c("numeric", "integer"),
        param_set = param_set,
        properties = properties,
        optimizer = optimizer,
        loss = loss,
        man = "mlr3torch::mlr_learners.test1"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nout = get_nout(task)
      nn_linear(length(task$feature_names), nout, bias = param_vals$bias)
    },
    .dataloader = function(task, param_vals) {
      ingress_token = TorchIngressToken(task$feature_names, batchgetter_num, c(NA, length(task$feature_names)))
      dataset = task_dataset(
        task,
        feature_ingress_tokens = list(num = ingress_token),
        target_batchgetter = crate(function(data, device) {
          torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
        }, .parent = topenv()),
        device = param_vals$device
      )
      dl_args = c(
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "num_workers",
        "collate_fn",
        "pin_memory",
        "drop_last",
        "timeout",
        "worker_init_fn",
        "worker_globals",
        "worker_packages"
      )
      args = param_vals[names(param_vals) %in% dl_args]
      invoke(dataloader, dataset = dataset)
    }
  )
)

LearnerTorchImageTest = R6Class("LearnerTorchImageTest",
  inherit = LearnerTorchImage,
  public = list(
    initialize = function(task_type, loss = t_loss("cross_entropy"), optimizer = t_opt("adam"), callbacks = list()) {
      param_set = ps(bias = p_lgl(tags = c("required", "train")))
      param_set$values = list(bias = FALSE)

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".image_test"),
        param_set = param_set,
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
    .network = function(task, param_vals) {
      shape = dd(task$data(task$row_ids[1L], task$feature_names)[[1L]])$pointer_shape
      d = prod(shape[-1])
      nout = get_nout(task)
      nn_sequential(
        nn_flatten(),
        nn_linear(d, nout, bias = param_vals$bias)
      )
    }
  )
)

classif_mlp2 = function() {
  l = LearnerTorchMLP$new("classif")
  l$param_set$set_values(epochs = 1L, batch_size = 100)
  l
}


regr_mlp2 = function() {
  l = LearnerTorchMLP$new("regr")
  l$param_set$set_values(epochs = 1L, batch_size = 100)
  l
}
