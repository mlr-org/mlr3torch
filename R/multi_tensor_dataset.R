multi_tensor_dataset = dataset("multi_tensor_dataset",
  initialize = function(dataset, device = "cpu") {
    assert_class(dataset, "dataset")
    # the return of dataset is list(x = list<torch_tensor>, y = torch_float, .index = torch_long)
    self$data = if (!is.null(dataset$.getbatch)) {
      dataset$.getbatch(seq_len(length(dataset)))
    } else {
      batches = lapply(seq_len(length(dataset)), dataset$.getitem)
      list(
        x = set_names(
          map(names(batches[[1]]$x), function(nm) torch_stack(map(batches, function(batch) batch$x[[nm]]))),
          nm = names(batches[[1]]$x)
        ),
        y  = torch_stack(map(batches, "y")),
        .index = do.call(c, map(batches, ".index"))
      )
    }
    self$data$x = lapply(self$data$x, function(x) x$to(device = device))
    self$data$y = self$data$y$to(device = device)
  },
  .getbatch = function(i) {
    list(
      x = map(self$data$x, function(x) x[i, drop = FALSE]),
      y = self$data$y[i, drop = FALSE],
      .index = self$data$.index[i, drop = FALSE]
    )
  },
  .length = function() {
    nrow(self$data$x[[1L]])
  }
)
