#' @title Data Descriptor
#'
#' @description
#' A data descriptor is a rather internal data structure used in the [`lazy_tensor`] data type.
#' In essence it is an annotated [`torch::dataset`] and a preprocessing graph (consisting mosty of [`PipeOpModule`]
#' operators). The additional meta data (e.g. pointer, shapes) allows to preprocess [`lazy_tensor`]s in an
#' [`mlr3pipelines::Graph`] just like any (non-lazy) data types.
#' The preprocessing is applied when [`materialize()`] is called on the [`lazy_tensor`].
#'
#' To create a data descriptor, you can also use the [`as_data_descriptor()`] function.
#'
#' @param dataset ([`torch::dataset`])\cr
#'   The torch dataset.
#'   It should return a named `list()` of [`torch_tensor`][torch::torch_tensor] objects.
#' @template param_dataset_shapes
#' @param graph ([`Graph`][mlr3pipelines::Graph])\cr
#'  The preprocessing graph.
#'  If left `NULL`, no preprocessing is applied to the data and `input_map`, `pointer`, `pointer_shape`, and
#' `pointer_shape_predict` are inferred in case the dataset returns only one element.
#' @param input_map (`character()`)\cr
#'   Character vector that must have the same length as the input of the graph.
#'   Specifies how the data from the `dataset` is fed into the preprocessing graph.
#' @param pointer (`character(2)` | `NULL`)\cr
#'   Points to an output channel within `graph`:
#'   Element 1 is the `PipeOp`'s id and element 2 is that `PipeOp`'s output channel.
#' @param pointer_shape (`integer()` | `NULL`)\cr
#'   Shape of the output indicated by `pointer`.
#' @param clone_graph (`logical(1)`)\cr
#'   Whether to clone the preprocessing graph.
#' @param pointer_shape_predict (`integer()` or `NULL`)\cr
#'   Internal use only.
#'   Used in a [`Graph`][mlr3pipelines::Graph] to anticipate possible mismatches between train and predict shapes.
#'
#' @details
#' While it would be more natural to define this as an S3 class, we opted for an R6 class to avoid the usual
#' trouble of serializing S3 objects.
#' If each row contained a DataDescriptor as an S3 class, this would copy the object when serializing.
#'
#' @export
#' @include utils.R
#' @seealso ModelDescriptor, lazy_tensor
#' @examplesIf torch::torch_is_installed()
#' # Create a dataset
#' ds = dataset(
#'   initialize = function() self$x = torch_randn(10, 3, 3),
#'   .getitem = function(i) list(x = self$x[i, ]),
#'   .length = function() nrow(self$x)
#' )()
#' dd = DataDescriptor$new(ds, list(x = c(NA, 3, 3)))
#' dd
#' # is the same as using the converter:
#' as_data_descriptor(ds, list(x = c(NA, 3, 3)))
DataDescriptor = R6Class("DataDescriptor",
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(dataset, dataset_shapes = NULL, graph = NULL, input_map = NULL, pointer = NULL,
      pointer_shape = NULL, pointer_shape_predict = NULL, clone_graph = TRUE) {
      assert_class(dataset, "dataset")
      assert_flag(clone_graph)
      # For simplicity we here require the first dimension of the shape to be NA so we don't have to deal with it,
      # e.g. during subsetting

      if (is.null(dataset_shapes)) {
        if (is.null(dataset$.getbatch)) {
          stopf("dataset_shapes must be provided if dataset does not have a `.getbatch` method.")
        }
        dataset_shapes = infer_shapes_from_getbatch(dataset)
      } else {
        assert_compatible_shapes(dataset_shapes, dataset)
      }

      if (is.null(graph)) {
        # avoid name conflicts
        if (is.null(input_map)) {
          assert_true(length(dataset_shapes) == 1L)
          input_map = names(dataset_shapes)
        }
        # get unique ID for input PipeOp
        graph = as_graph(po("nop", id =
          paste0("nop.", substr(calculate_hash(address(dataset)), 1, 6), ".", input_map)
        ), clone = FALSE)
      } else {
        graph = as_graph(graph, clone = clone_graph)
        assert_true(length(graph$pipeops) >= 1L)
      }
      # no preprocessing, dataset returns only a single element (there we can infer a lot)
      simple_case = length(graph$pipeops) == 1L && inherits(graph$pipeops[[1L]], "PipeOpNOP") &&
        length(dataset_shapes) == 1L

      if (is.null(input_map) && nrow(graph$input) == 1L && length(dataset_shapes) == 1L) {
        input_map = names(dataset_shapes)
      } else {
        assert_subset(input_map, names(dataset_shapes))
      }
      if (is.null(pointer) && nrow(graph$output) == 1L) {
        pointer = c(graph$output$op.id, graph$output$channel.name)
      } else {
        assert_character(pointer, len = 2L)
        assert_choice(pointer[[1]], names(graph$pipeops))
        assert_choice(pointer[[2]], graph$pipeops[[pointer[[1]]]]$output$name)
      }
      if (is.null(pointer_shape) && simple_case) {
        pointer_shape = dataset_shapes[[1L]]
      } else {
        assert_shape(pointer_shape, null_ok = TRUE)
      }
      if (is.null(pointer_shape_predict) && simple_case) {
        pointer_shape_predict = pointer_shape
      } else if (simple_case) {
        assert_true(isTRUE(all.equal(pointer_shape, pointer_shape_predict)))
      } else {
        assert_shape(pointer_shape_predict, null_ok = TRUE)
      }

      assert_subset(paste0(pointer, collapse = "."), graph$output$name)
      assert_true(length(input_map) == length(graph$input$name))

      # We hash the address of the environment, because the hashes of an environment are not stable,
      # even with a .dataset (that should usually not really have a state), hashes might change due to byte-code
      # compilation
      dataset_hash = calculate_hash(address(dataset))

      self$dataset = dataset
      self$graph = graph
      self$dataset_shapes = dataset_shapes
      self$input_map = input_map
      self$pointer = pointer
      self$pointer_shape = pointer_shape
      self$dataset_hash = dataset_hash
      self$graph_input = graph$input$name
      self$pointer_shape_predict = pointer_shape_predict

      # the pointer is not taken into account, because we save the whole output during caching (for which the
      # hash is primarily used)
      self$hash = calculate_hash(self$dataset_hash, self$graph$hash, self$input_map)
    },
    #' @description Prints the object
    #' @param ... (any)\cr
    #' Unused
    print = function(...) {
      catn(sprintf("<DataDescriptor: %d ops>", length(self$graph$pipeops)))
      catn(sprintf("* dataset_shapes: %s", shape_to_str(self$dataset_shapes)))
      catn(sprintf("* input_map: (%s) -> Graph", paste0(self$input_map, collapse = ", ")))
      catn(sprintf("* pointer: %s", paste0(self$pointer, collapse = ".")))
      catn(str_indent("* shape:",
        if (is.null(self$pointer_shape)) "<unknown>" else shape_to_str(list(self$pointer_shape))))
      # dont print the predict shape, it is too confusing
    },
    #' @field dataset ([`torch::dataset`])\cr
    #' The dataset.
    dataset = NULL,
    #' @field graph ([`Graph`][mlr3pipelines::Graph])\cr
    #' The preprocessing graph.
    graph = NULL,
    #' @field dataset_shapes (named `list()` of (`integer()` or `NULL`))\cr
    #' The shapes of the output.
    dataset_shapes = NULL,
    #' @field input_map (`character()`)\cr
    #' The input map from the dataset to the preprocessing graph.
    input_map = NULL,
    #' @field pointer (`character(2)`)\cr
    #' The output pointer.
    pointer = NULL,
    #' @field pointer_shape (`integer()` | `NULL`)\cr
    #' The shape of the output indicated by `pointer`.
    pointer_shape = NULL,
    #' @field dataset_hash (`character(1)`)\cr
    #' Hash for the wrapped dataset.
    dataset_hash = NULL,
    #' @field hash (`character(1)`)\cr
    #' Hash for the data descriptor.
    hash = NULL,
    #' @field graph_input (`character()`)\cr
    #' The input channels of the preprocessing graph (cached to save time).
    graph_input = NULL,
    #' @field pointer_shape_predict (`integer()` or `NULL`)\cr
    #' Internal use only.
    pointer_shape_predict = NULL
  )
)

#' @title Convert to Data Descriptor
#' @description
#' Converts the input to a [`DataDescriptor`].
#' @param x (any)\cr
#'   Object to convert.
#' @template param_dataset_shapes
#' @param ... (any)\cr
#'   Further arguments passed to the [`DataDescriptor`] constructor.
#' @export
#' @examplesIf torch::torch_is_installed()
#' ds = dataset("example",
#'   initialize = function() self$iris = iris[, -5],
#'   .getitem = function(i) list(x = torch_tensor(as.numeric(self$iris[i, ]))),
#'   .length = function() nrow(self$iris)
#' )()
#' as_data_descriptor(ds, list(x = c(NA, 4L)))
#'
#' # if the dataset has a .getbatch method, the shapes are inferred
#' ds2 = dataset("example",
#'   initialize = function() self$iris = iris[, -5],
#'   .getbatch = function(i) list(x = torch_tensor(as.matrix(self$iris[i, ]))),
#'   .length = function() nrow(self$iris)
#' )()
#' as_data_descriptor(ds2)
as_data_descriptor = function(x, dataset_shapes, ...) {
  UseMethod("as_data_descriptor")
}

#' @export
as_data_descriptor.dataset = function(x, dataset_shapes = NULL, ...) {
  DataDescriptor$new(x, dataset_shapes = dataset_shapes, ...)
}

infer_shapes_from_getbatch = function(ds) {
  example = ds$.getbatch(1L)
  if (!test_list(example, names = "unique", types = "torch_tensor")) {
    stopf("Dataset must return a named list of tensors, but it does not")
  }
  map(example, function(x) {
    shape = x$shape
    shape[1L] = NA
    shape
  })
}

assert_compatible_shapes = function(shapes, dataset) {
  assert_shapes(shapes, null_ok = TRUE, unknown_batch = TRUE, named = TRUE)

  # prevent user from e.g. forgetting to wrap the return in a list
  example = if (is.null(dataset$.getbatch)) {
    dataset$.getitem(1L)
  } else {
    dataset$.getbatch(1L)
  }
  if (!test_list(example, names = "unique") || !test_permutation(names(example), names(shapes))) {
    stopf("Dataset must return a list with named elements that are a permutation of the dataset_shapes names.")
  }
  iwalk(example, function(x, nm) {
    if (!test_class(x, "torch_tensor")) {
      stopf("The dataset must return torch tensors, but element '%s' is of class %s", nm, class(x)[[1L]])
    }
  })

  if (is.null(dataset$.getbatch)) {
    example = map(example, function(x) x$unsqueeze(1))
  }

  iwalk(shapes, function(dataset_shape, name) {
    if (!is.null(dataset_shape) && !test_equal(shapes[[name]][-1], example[[name]]$shape[-1L])) {
      expected_shape = example[[name]]$shape
      expected_shape[1] = NA
      stopf(paste0("First batch from dataset is incompatible with the provided shape of %s:\n",
        "* Provided shape: %s.\n* Expected shape: %s."), name,
        shape_to_str(unname(shapes[name])), shape_to_str(list(expected_shape)))
    }
  })
}
