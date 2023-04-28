# FIXME: maybe export?
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

dataset_img = function(self, task, param_vals) {
  assert_true(length(task$feature_names) == 1)
  # TODO: Maybe we want to be more careful here to avoid changing parameters between train and predict
  imgshape = c(param_vals$channels, param_vals$height, param_vals$width)

  batchgetter = batchgetter_img(imgshape)


  ingress_tokens = list(image = TorchIngressToken(task$feature_names, batchgetter, imgshape))

  task_dataset(
    task,
    feature_ingress_tokens = ingress_tokens,
    target_batchgetter = crate(function(data, device) {
      torch_tensor(data = as.integer(data[[1]]), dtype = torch_long(), device = device)
    }, .parent = topenv()),
    device = param_vals$device
  )
}

dataset_num = function(self, task, param_vals) {
  num_features = task$feature_types[get("type") %in% c("numeric", "integer"), "id"][[1L]]
  ingress = TorchIngressToken(num_features, batchgetter_num, c(NA, length(task$feature_names)))

  task_dataset(
    task,
    feature_ingress_tokens = list(input = ingress),
    target_batchgetter = target_batchgetter(task$task_type)
  )
}

dataset_num_categ = function(self, task, param_vals) {
  features_num = task$feature_types[get("type") %in% c("numeric", "integer"), "id"][[1L]]
  features_categ = task$feature_types[get("type") %in% c("factor", "ordered"), "id"][[1L]]

  tokens = list()

  if (length(features_num)) {
    tokens$input_num = TorchIngressToken(features_num, batchgetter_num, c(NA, length(features_num)))
  }
  if (length(features_categ)) {
    tokens$input_categ = TorchIngressToken(features_categ, batchgetter_categ, c(NA, length(features_categ)))
  }

  assert_true(length(tokens) >= 1)

  task_dataset(
    task,
    feature_ingress_tokens = tokens,
    target_batchgetter = target_batchgetter(task$task_type)
  )
}

batchgetter_img = function(imgshape) {
  crate(function(data, device) {
    tensors = lapply(data[[1]], function(uri) {
      tnsr = torchvision::transform_to_tensor(magick::image_read(uri))
      assert_true(identical(tnsr$shape, imgshape))
      torch_reshape(tnsr, imgshape)$unsqueeze(1)
    })
    torch_cat(tensors, dim = 1)$to(device = device)
  }, imgshape, .parent = topenv())
}

target_batchgetter = function(task_type) {
  if (task_type == "classif") {
    target_batchgetter = (function(data, device) {
      torch_tensor(data = as.integer(data[[1L]]), dtype = torch_long(), device)
    })
  } else if (task_type == "regr") {
    target_batchgetter = crate(function(data, device) {
      torch_tensor(data = data[[1L]], dtype = torch_float32(), device)
    })
  } else {
    stopf("Unsupported task type %s", task_type)
  }

  return(target_batchgetter)
}
