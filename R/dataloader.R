task_dataset = dataset(
  initialize = function(task, feature_ingress_tokens, target_batchgetter = NULL, device = "cpu") {
    self$task = assert_r6(task, "Task")
    self$feature_ingress_tokens = assert_list(feature_ingress_tokens, types = "TorchIngressToken", names = "unique")
    self$all_features = unique(c(unlist(map(feature_ingress_tokens, "features")), task$target_names))
    assert_subset(self$all_features, c(task$target_names, task$feature_names))
    self$target_batchgetter = assert_function(target_batchgetter, args = c("data", "device"), null.ok = TRUE)
    self$device = device
  },
  .getbatch = function(index) {
    datapool = self$task$data(rows = self$task$row_ids[index], cols = self$all_features)
    x = lapply(self$feature_ingress_tokens, function(it) {
      it$batchgetter(datapool[, it$features, with = FALSE], self$device)
    })
    y = if (!is.null(self$target_batchgetter)) {
      self$target_batchgetter(datapool[, self$task$target_names, with = FALSE],
        self$device)
    }
    list(x = x, y = y, .index = index)
  },
  .length = function() {
    self$task$nrow
  }
)

learner_torch_classif_dataloader = function(task, param_vals, feature_ingress_tokens) {
  dataset = task_dataset(
    task,
    feature_ingress_tokens = feature_ingress_tokens,
    target_batchgetter = crate(function(data, device) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
    }, .parent = topenv()),
    device = param_vals$device
  )
  dataloader(
    dataset = dataset,
    batch_size = param_vals$batch_size,
    drop_last = param_vals$drop_last,
    shuffle = param_vals$shuffle
  )
}



.get_batchgetter = function(task, param_vals) {
  imgshape = c(param_vals$channels, param_vals$height, param_vals$width)
  crate(function(data, device) {
    tensors = lapply(data[[1]], function(uri) {
      tnsr = torchvision::transform_to_tensor(magick::image_read(uri))
      assert_true(identical(tnsr$shape, imgshape))
      torch_reshape(tnsr, imgshape)$unsqueeze(1)
    })
    torch_cat(tensors, dim = 1)$to(device = device)
  }, imgshape, .parent = topenv())
}
