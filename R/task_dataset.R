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
#'
#' @param task ([`Task`])\cr
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
  initialize = function(task, feature_ingress_tokens, target_batchgetter = NULL, device) {
    self$task = assert_r6(task$clone(deep = TRUE), "Task")
    iwalk(feature_ingress_tokens, function(it, nm) {
      if (length(it$features) == 0) {
        stopf("Received ingress token '%s' with no features.", nm)
      }
    })
    self$feature_ingress_tokens = assert_list(feature_ingress_tokens, types = "TorchIngressToken", names = "unique")
    self$all_features = unique(c(unlist(map(feature_ingress_tokens, "features")), task$target_names))
    assert_subset(self$all_features, c(task$target_names, task$feature_names))
    self$target_batchgetter = assert_function(target_batchgetter, args = c("data", "device"), null.ok = TRUE)
    self$device = assert_choice(device, mlr_reflections$torch$devices)

    lazy_tensor_features = self$task$feature_types[get("type") == "lazy_tensor"][[1L]]

    data = self$task$data(cols = lazy_tensor_features)

    # Here, we could have multiple `lazy_tensor` columns that share parts of the graph
    # We try to merge those graphs if possible
    if (length(lazy_tensor_features) > 1L) {
      merge_result = try(merge_lazy_tensor_graphs(data), silent = TRUE)

      if (inherits(merge_result, "try-error")) {
        # This should basically never happen
        lg$debug("Failed to merge data descriptor, this might lead to inefficient preprocessing.")
        # TODO: test that is still works when this triggers
      } else {
        self$task$cbind(merge_result)
      }
    }

    # we can cache the output (hash) or the data (dataset_hash)
    self$cache_lazy_tensors = length(unique(map_chr(data, function(x) dd(x)$hash))) > 1L ||
      length(unique(map_chr(data, function(x) dd(x)$dataset_hash))) > 1L

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
    list(x = x, y = y, .index = index)
  },
  .length = function() {
    self$task$nrow
  }
)

optimize_lazy_tensors = function(lts) {
}

merge_lazy_tensors = function(lts) {
  dataset = make_dataset_collection(lts)
  graph = Reduce(merge_graphs, map(lts, function(x) dd(x)$graph))

  input_map = Reduce(c, map(lts, function(lt) {
    set_names(list(dd(lt)$input_map), dd(lt)$graph$input$name)
  }))
  input_map = input_map[unique(names(input_map))]
  input_map = unname(unlist(input_map[graph$input$name]))

  # some PipeOs that were previously terminal might not be anymore,
  # for those we add nops and updaate the pointers for their data descriptors
  map_dtc(lts, function(lt) {
    pointer_name = paste0(dd(lt)$pointer, collapse = ".")

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
      dataset = dd(lts[[1]])$dataset,
      dataset_shapes = dd(lts[[1L]])$dataset_shapes,
      graph = graph,
      input_map = dd(lts[[1]])$input_map,
      pointer = pointer,
      pointer_shape = dd(lt)$pointer_shape,
      pointer_shape_predict = dd(lt)$pointer_shape_predict,
      clone_graph = FALSE
    )
    new_lazy_tensor(data_descriptor, map_int(lt, 1L))
  })

  # 1) Merge the datasets

  # 2) Merge the graphs
  # 3) Adjust the input maps


  names_lts = names(lts)
  # we onl attempt to merge preprocessing graphs that have the same dataset_hash
  groups = map_chr(lts, function(lt) dd(lt)$dataset_hash)
  lts = unlist(map(unique(groups), function(group) {
    lts_to_merge = lts[, names_lts[groups == group, with = FALSE]]
    if (length(lts_to_merge) > 1L) {
      merge_compatible_lazy_tensor_graphs(lts_to_merge)
    } else {
      lts_to_merge
    }
  }), recursive = FALSE)

  as_data_backend(as.data.table(set_names(lts, names_lts)))
}

make_dataset_collection_batch = function(lts) {
  dataset("dataset_collection_batch",
    initialize = function(lts) {
      datasets = map(lts, function(lt) dd(lt)$dataset)
      duplicated = duplicated(map(lts, function(lt) dd(lt)$dataset_hash))
      self$datasets = datasets[!duplicated]
      self$ids = map(lts, function(lt) map(lt, 1L))[!duplicated]
      self$names = Reduce(c, map(lts[!duplicated], function(lt) {
        paste0(substring(lt$dataset_hash, 1, 5), ".", names(dd(lt)$dataset_shapes))
      }))
    },
    .getbatch = function(i) {
      batch = list()
      for (j in seq_along(self$datasets)) {
        idx = self$ids[[j]][i]
        batch = append(batch, self$datasets[[j]]$.getbatch(idx))
      }
      set_names(batch, self$names)
    }
  )
}
make_dataset_collection_item = function(lts) {
  dataset("dataset_collection_batch",
    initialize = function(lts) {
      datasets = map(lts, function(lt) dd(lt)$dataset)
      duplicated = duplicated(map(lts, function(lt) dd(lt)$dataset_hash))
      self$datasets = datasets[!duplicated]
      self$ids = map(lts, function(lt) map(lt, 1L))[!duplicated]
      self$names = Reduce(c, map(lts[!duplicated], function(lt) {
        paste0(substring(lt$dataset_hash, 1, 5), ".", names(dd(lt)$dataset_shapes))
      }))
    },
    .getitem = function(i) {
      batch = list()
      for (j in seq_along(self$datasets)) {
        idx = self$ids[[j]][i]
        if (is.null(self$dataset[[j]]$.getbatch)) {
          new = self$datasets[[j]]$.getitem(idx)
        } else {
          new = map(self$datasets[[j]]$.getitem(idx), function(x) x$squeeze(1L))
        }
        batch = append(batch, new)
      }
      set_names(batch, self$names)
    }
  )(lts)
}

make_dataset_collection = function(lts) {
  # in principle it can now happen, that we cannot use the .getbatch method of
  assert_true(length(unique(lengths(lts))) == 1L)
  if (any(map_lgl(lts, function(lt) is.null(dd(lt)$dataset$.getbatch)))) {
    make_dataset_collection_item(lts)
  } else {
    make_dataset_collection_batch(lts)
  }
}

merge_lazy_tensors = function(lts) {

}

merge_compatible_lazy_tensor_graphs = function(lts) {

}

dataset_ltnsr = function(task, param_vals) {
  assert_true(length(task$feature_names) == 1L)
  shape = dd(task$data(cols = task$feature_names)[[1L]])$pointer_shape
  ingress_tokens = list(image = TorchIngressToken(task$feature_names, batchgetter_lazy_tensor, shape))

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
