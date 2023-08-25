#' @title Create a Dataset from a Task
#'
#' @description
#' Creates a torch [dataset][torch::dataset] from an mlr3 [`Task`].
#' The resulting dataset's `$.get_batch()` method returns a list with elements `x`, `y` and `index`:
#' * `x` is a list with tensors, whose content is defined by the parameter `feature_ingress_tokens`.
#' * `y` is the target variable and its content is defined by the parameter `target_batchgetter`.
#' * `.index` is the index of the batch in the task's data.
#'
#' The data is returned on the device specified by the parameter `device`.
#' FIXME: This is now not necessarily true (because of the lazy_tensor column.)
#'
#' @param task ([`Task`])\cr
#'   The task for which to build the [dataset][torch::dataset].
#' @param feature_ingress_tokens (named `list()` of [`TorchIngressToken`])\cr
#'   Each ingress token defines one item in the `$x` value of a batch with corresponding names.
#' @param target_batchgetter (`function(data, device)`)\cr
#'   A function taking in arguments `data`, which is a `data.table` containing only the target variable, and `device`.
#'   It must return the target as a torch [tensor][torch::torch_tensor] on the selected device.
#' @param device (`character()`)\crZYZ
#'   The device, e.g. `"cuda"` or `"cpu"`.
#' @export
#' @return [`torch::dataset`]
#' @examples
#' task = tsk("iris")
#' sepal_ingress = TorchIngressToken(
#'   features = c("Sepal.Length", "Sepal.Width"),
#'   batchgetter = batchgetter_num,
#'   shape = c(NA, 2)
#' )
#' petal_ingress = TorchIngressToken(
#'   features = c("Petal.Length", "Petal.Width"),
#'   batchgetter = batchgetter_num,
#'   shape = c(NA, 2)
#' )
#' ingress_tokens = list(sepal = sepal_ingress, petal = petal_ingress)
#'
#' target_batchgetter = function(data, device) {
#'   torch_tensor(data = data[[1L]], dtype = torch_float32(), device)$unsqueeze(2)
#' }
#' dataset = task_dataset(task, ingress_tokens, target_batchgetter, "cpu")
#' batch = dataset$.getbatch(1:10)
#' batch
task_dataset = dataset(
  initialize = function(task, feature_ingress_tokens, target_batchgetter = NULL, device, preprocessor = NULL) {
    self$task = assert_r6(task$clone(deep = TRUE), "Task")
    self$feature_ingress_tokens = assert_list(feature_ingress_tokens, types = "TorchIngressToken", names = "unique")
    self$preprocessor = preprocessor

    iwalk(feature_ingress_tokens, function(it, nm) {
      if (length(it$features) == 0) {
        stopf("Received ingress token '%s' with no features.", nm)
      }
    })

    # we already calculate all the unique datasets for each row here.

    self$all_features = unique(c(unlist(map(feature_ingress_tokens, "features")), task$target_names))

    tmp = task$data(task$row_ids[1], self$all_features)

    self$tensor_columns = names(tmp)[map_lgl(tmp, function(x) test_class(x, "lazy_tensor"))]
    assert_subset(self$all_features, c(task$target_names, task$feature_names))
    self$target_batchgetter = assert_function(target_batchgetter, args = c("data", "device"), null.ok = TRUE)
    self$device = assert_choice(device, mlr_reflections$torch$devices)
  },
  .getbatch = function(index) {
    datapool = self$task$data(rows = self$task$row_ids[index], cols = self$all_features)

    # we have (id, col, torch_loader) pairs, but we want to load every (id, torch_loader) pair only once for efficiency

    tensor_env = new.env()
    datapool_lazy_resolved = map_dtc(datapool[, self$tensor_columns, with = FALSE], function(column) {
      map(column, function(elt) {
        # elt[[1]] is the id
        # elt[[2]] is the column (not used yet)
        # elt [[3]] is the dataset_descriptor (to rename)
        torch_dataset = elt$torch_dataset
        hash = torch_dataset$hash
        id = elt$id
        # we cache the `$load()` result for usage afterwards
        if (!exists(hash, tensor_env)) {
          tensor_env[[hash]] = list()
          tensor_env[[hash]][[id]] = torch_dataset$get(id)
        } else if (is.null(tensor_env[[hash]][[id]])) {
          tensor_env[[hash]][[id]] = torch_dataset$get(id)
        }

        tensor_env[[hash]][[id]][[elt$column]]
      })
    })

    datapool_lazy_resolved = set_names(datapool_lazy_resolved, self$tensor_columns)
    datapool[, self$tensor_columns] = NULL
    datapool = cbind(datapool, datapool_lazy_resolved)

    x = lapply(self$feature_ingress_tokens, function(it) {
      it$batchgetter(datapool[, it$features, with = FALSE], self$device)
    })
    y = if (!is.null(self$target_batchgetter)) {
      self$target_batchgetter(datapool[, self$task$target_names, with = FALSE],
        self$device)
    }

    if (is.null())


    list(x = x, y = y, .index = index)
  },
  .length = function() {
    self$task$nrow
  }
)

dataset_img = function(task, param_vals) {
  # TODO: Maybe we want to be more careful here to avoid changing parameters between train and predict
  # Instead use the param vals stored in the state?
  imgshape = c(param_vals$channels, param_vals$height, param_vals$width)

  batchgetter = get_batchgetter_img(imgshape)

  ingress_tokens = list(image = TorchIngressToken(task$feature_names, batchgetter, imgshape))

  task_dataset(
    task,
    feature_ingress_tokens = ingress_tokens,
    target_batchgetter = get_target_batchgetter(task$task_type),
    device = param_vals$device
  )
}

dataset_num = function(task, param_vals) {
  num_features = task$feature_types[get("type") %in% c("numeric", "integer"), "id"][[1L]]
  ingress = TorchIngressToken(num_features, batchgetter_num, c(NA, length(task$feature_names)))

  task_dataset(
    task,
    feature_ingress_tokens = list(input = ingress),
    target_batchgetter = get_target_batchgetter(task$task_type),
    device = param_vals$device
  )
}

dataset_num_categ = function(task, param_vals) {
  features_num = task$feature_types[get("type") %in% c("numeric", "integer"), "id"][[1L]]
  features_categ = task$feature_types[get("type") %in% c("factor", "ordered", "logical"), "id"][[1L]]

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
    target_batchgetter = get_target_batchgetter(task$task_type),
    device = param_vals$device
  )
}


#' @title Batchgetter for Numeric Data
#'
#' @description
#' Converts a data frame of numeric data into a float tensor by calling `as.matrix()`.
#' No input checks are performed
#'
#' @param data (`data.table()`)\cr
#'   `data.table` to be converted to a `tensor`.
#' @param device (`character(1)`)\cr
#'   The device on which the tensor should be created.
#' @export
batchgetter_num = function(data, device) {
  torch_tensor(
    data = as.matrix(data),
    dtype = torch_float(),
    device = device
  )
}


#' @title Batchgetter for Categorical data
#'
#' @description
#' Converts a data frame of categorical data into a long tensor by converting the data to integers.
#' No input checks are performed.
#'
#' @param data (`data.table`)\cr
#'   `data.table` to be converted to a `tensor`.
#' @param device (`character(1)`)\cr
#'   The device.
#' @export
batchgetter_categ = function(data, device) {
  torch_tensor(
    data = as.matrix(data[, lapply(.SD, as.integer)]),
    dtype = torch_long(),
    device = device
  )
}

get_batchgetter_img = function(imgshape) {
  crate(function(data, device) {
    tensors = lapply(data[[1]], function(uri) {
      tnsr = torchvision::transform_to_tensor(magick::image_read(uri))
      assert_true(length(tnsr$shape) == length(imgshape) && all(tnsr$shape == imgshape))
      torch_reshape(tnsr, imgshape)$unsqueeze(1)
    })
    torch_cat(tensors, dim = 1)$to(device = device)
  }, imgshape, .parent = topenv())
}

target_batchgetter_classif = function(data, device) {
  torch_tensor(data = as.integer(data[[1L]]), dtype = torch_long(), device = device)
}

target_batchgetter_regr = function(data, device) {
  torch_tensor(data = data[[1L]], dtype = torch_float32(), device = device)$unsqueeze(2)
}

get_target_batchgetter = function(task_type) {
  switch(task_type,
    classif = target_batchgetter_classif,
    regr = target_batchgetter_regr
  )
}
