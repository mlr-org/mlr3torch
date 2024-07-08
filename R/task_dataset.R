#' @title Create a Dataset from a Task
#'
#' @description
#' Creates a torch [dataset][torch::dataset] from an mlr3 [`Task`][mlr3::Task].
#' The resulting dataset's `$.get_batch()` method returns a list with elements `x`, `y` and `index`:
#' * `x` is a list with tensors, whose content is defined by the parameter `feature_ingress_tokens`.
#' * `y` is the target variable and its content is defined by the parameter `target_batchgetter`.
#' * `.index` is the index of the batch in the task's data.
#'
#' The data is returned on the device specified by the parameter `device`.
#'
#' @param task ([`Task`][mlr3::Task])\cr
#'   The task for which to build the [dataset][torch::dataset].
#' @param feature_ingress_tokens (named `list()` of [`TorchIngressToken`])\cr
#'   Each ingress token defines one item in the `$x` value of a batch with corresponding names.
#' @param target_batchgetter (`function(data, device)`)\cr
#'   A function taking in arguments `data`, which is a `data.table` containing only the target variable, and `device`.
#'   It must return the target as a torch [tensor][torch::torch_tensor] on the selected device.
#' @param device (`character()`)\cr
#'   The device, e.g. `"cuda"` or `"cpu"`.
#' @export
#' @return [`torch::dataset`]
#' @examplesIf torch::torch_is_installed()
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
  initialize = function(task, feature_ingress_tokens, target_batchgetter = NULL, device) {
    self$task = assert_r6(task$clone(deep = TRUE), "Task")
    iwalk(feature_ingress_tokens, function(it, nm) {
      if (length(it$features) == 0) {
        stopf("Received ingress token '%s' with no features.", nm)
      }
    })
    assert_list(feature_ingress_tokens, types = "TorchIngressToken", names = "unique", min.len = 1L)
    self$feature_ingress_tokens = assert_list(feature_ingress_tokens, types = "TorchIngressToken", names = "unique")
    self$all_features = unique(c(unlist(map(feature_ingress_tokens, "features")), task$target_names))
    assert_subset(self$all_features, c(task$target_names, task$feature_names))
    self$target_batchgetter = assert_function(target_batchgetter, args = c("data", "device"), null.ok = TRUE)
    self$device = assert_choice(device, mlr_reflections$torch$devices)

    lazy_tensor_features = self$task$feature_types[get("type") == "lazy_tensor"][[1L]]

    data = self$task$data(cols = lazy_tensor_features)

    # Here, we could have multiple `lazy_tensor` columns that share parts of the graph
    # we only try to merge those with the same DataDescriptor, this is a restriction which we might want to
    # relax later, but it eases data-loading
    if (length(lazy_tensor_features) > 1L) {
      merged_cols = merge_lazy_tensor_graphs(data)
      if (!is.null(merged_cols)) {
        self$task$cbind(merged_cols)
      }
    }

    self$cache_lazy_tensors = auto_cache_lazy_tensors(data)
  },
  .getbatch = function(index) {
    cache = if (self$cache_lazy_tensors) new.env()

    datapool = self$task$data(rows = self$task$row_ids[index], cols = self$all_features)
    x = lapply(self$feature_ingress_tokens, function(it) {
      it$batchgetter(datapool[, it$features, with = FALSE], self$device, cache = cache)
    })

    y = if (!is.null(self$target_batchgetter)) {
      self$target_batchgetter(datapool[, self$task$target_names, with = FALSE],
        self$device)
    }
    out = list(x = x, .index = torch_tensor(index, device = self$device, dtype = torch_long()))
    if (!is.null(y)) out$y = y
    return(out)
  },
  .length = function() {
    self$task$nrow
  }
)

# This returns a data.table or NULL (if nothing was merged)
merge_lazy_tensor_graphs = function(lts) {
  # we only attempt to merge preprocessing graphs that have the same dataset_hash
  hashes = map_chr(lts, function(lt) dd(lt)$dataset_hash)
  # we remove columns that don't share dataset_hash with other columns
  hashes_to_merge = unique(hashes[duplicated(hashes)])
  lts = lts[, hashes %in% hashes_to_merge, with = FALSE]
  hashes_subset = map_chr(lts, function(lt) dd(lt)$dataset_hash)

  names_lts = names(lts)
  lts = map(hashes_to_merge, function(hash) {
    x = try(merge_compatible_lazy_tensor_graphs(lts[, names_lts[hashes_subset == hash], with = FALSE]), silent = TRUE)
    if (inherits(x, "try-error")) {
      lg$warn("Cannot merge lazy tensors with data descriptor with hash '%s'", hash)
      return(NULL)
    }
    x
  })

  Reduce(cbind, lts)
}

merge_compatible_lazy_tensor_graphs = function(lts) {
  # all inputs have the same dataset (makes data-loading easier)
  graph = Reduce(merge_graphs, map(lts, function(x) dd(x)$graph))

  # now we need to calculate the new input map, some of the graphs that were merged have different,
  # others the same input pipeops
  input_map = Reduce(c, map(lts, function(lt) {
    set_names(list(dd(lt)$input_map), dd(lt)$graph$input$name)
  }))
  # all input pipeops that share IDs must be identical and hence receive the same data,
  # therefore we can remove duplicates
  input_map = input_map[unique(names(input_map))]
  input_map = unname(unlist(input_map[graph$input$name]))

  map_dtc(lts, function(lt) {
    pointer_name = paste0(dd(lt)$pointer, collapse = ".")
    # some PipeOs that were previously terminal might not be anymore,
    # for those we add nops and update the pointers for their data descriptors
    pointer = if (pointer_name %nin% graph$output$name) {
      po_terminal = po("nop", id = uniqueify(pointer_name, graph$ids()))
      graph$add_pipeop(po_terminal, clone = FALSE)
      graph$add_pipeop(
        src_id = dd(lt)$pointer[1L],
        dst_id = po_terminal$id,
        src_channel = dd(lt)$pointer[2L],
        dst_channel = po_terminal$input$name
      )

      c(po_terminal$id, po_terminal$output$name)
    } else {
      dd(lt)$pointer
    }

    data_descriptor = DataDescriptor$new(
      dataset = dd(lts[[1]])$dataset, # all are the same
      dataset_shapes = dd(lts[[1L]])$dataset_shapes, # all are the same
      graph = graph, # was merged
      input_map = input_map, # was merged
      pointer = pointer, # is set per lt
      pointer_shape = dd(lt)$pointer_shape, # is set per lt
      pointer_shape_predict = dd(lt)$pointer_shape_predict, # is set per lt
      clone_graph = FALSE # shallow clone already created above
    )
    new_lazy_tensor(data_descriptor, map_int(lt, 1L))
  })
}

dataset_ltnsr = function(task, param_vals) {
  po_ingress = po("torch_ingress_ltnsr", shape = param_vals$shape)
  md = po_ingress$train(list(task))[[1L]]
  ingress = md$ingress
  names(ingress) = "input"
  task_dataset(
    task = task,
    feature_ingress_tokens = ingress,
    target_batchgetter = get_target_batchgetter(task$task_type),
    device = param_vals$device
  )
}

dataset_num = function(task, param_vals) {
  po_ingress = po("torch_ingress_num")
  md = po_ingress$train(list(task))[[1L]]
  ingress = md$ingress
  names(ingress) = "input"
  task_dataset(
    task = task,
    feature_ingress_tokens = md$ingress,
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
#' @param ... (any)\cr
#'   Unused.
#' @export
batchgetter_num = function(data, device, ...) {
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
#' @param ... (any)\cr
#'   Unused.
#' @export
batchgetter_categ = function(data, device, ...) {
  torch_tensor(
    data = as.matrix(data[, lapply(.SD, as.integer)]),
    dtype = torch_long(),
    device = device
  )
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
