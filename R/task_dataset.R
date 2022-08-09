task_dataset = dataset(
  initialize = function(task, feature_ingress_tokens, target_batchgetter = NULL, device) {
    self$task = assert_r6(task, "Task")
    self$ingress_tokens = assert_list(ingress_tokens, types = "TorchIngressToken", names = "unique")
    self$all_features = unique(c(unlist(map(ingress_tokens, "features")), task$target_names))
    assert_subset(self$all_features, c(task$target_names, task$feature_names))
    self$target_batchgetter = assert_function(target_batchgetter, args = c("data", "device"), null.ok = TRUE)
    self$device = device
  },
  .getbatch = function(index) {
    datapool = self$task$data(rows = self$task$row_ids[index], cols = self$all_features)
    x = lapply(self$ingress_tokens, function(it) {
      it$batchgetter(datapool[, it$features, with = FALSE], self$device)
    })
    y = if (!is.null(self$target_batchgetter)) self$target_batchgetter(datapool[, sekf$task$target_names, with = FALSE], self$device)
    list(x = x, y = y)
  },
  .length = function() {
    self$task$nrow
  }
)
