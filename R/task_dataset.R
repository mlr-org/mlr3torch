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
    y = if (!is.null(self$target_batchgetter)) self$target_batchgetter(datapool[, self$task$target_names, with = FALSE], self$device)
    list(x = x, y = y, .index = index)
  },
  .length = function() {
    self$task$nrow
  }
)
