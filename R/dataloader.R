# FIXME: Do we really need this function (after decomposition into dl and ds method?)
dataloader_num = function(self, task, param_vals) {
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
