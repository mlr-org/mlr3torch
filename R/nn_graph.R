#' @title Graph Network
#' @description
#' NOTE: Don't use this directly but build it using [TorchOp]s.
#'
#' Neural networks can be represented as Graphs as defined in [mlr3pipelines].
#' The nodes define the operations that are being applied to the data, and the edges define
#' the data-flow.
#' An graph network essentially consists of a list of `nn_module`s
#'
#'
#'
#'
#'
#' @examples
#' \dontrun{
#' # define a graph that represents a skip-connection
#' graph = top("input") %>>%
#'   gunion(
#'     list(
#'       top("linear_1", out_features = 10L) %>>% top("relu"),
#'       top("linear_2")
#'     )
#'   ) %>>%
#'   top("add") %>>%
#'   top("output")
#'
#' graph$plot()
#' }
# '@export
nn_graph = nn_module(
  "nn_graph",
  initialize = function() {
    # TODO: maybe check for topological sort?
    private$.edges = setDT(named_list(c("src_id", "dst_id", "src_channel", "dst_channel"), character(0)))
    private$.ids = character(0)
    private$.outputs = list()
    private$.inputs = list()
    # edges = edges[order(dst_channel), list(src_channel, dst_id, dst_channel, payload), by = src_id]
  },
  set_terminal = function() {
    last_id = tail(private$.ids, 1L)
    last_channel = private$.outputs[[last_id]]
    assert_true(length(last_channel) == 1L)
    self$add_edge(
      src_id = last_id,
      dst_id = "__terminal__",
      src_channel = last_channel,
      dst_channel = "output"
    )
    invisible(self)
  },
  forward = function(input) {
    edges = private$.edges
    # loads the input into all channels
    edges["__initial__", payload := list(list(..input)), on = "src_id"]
    for (id in private$.ids) {
      layer = self[[id]]
      input_name = private$.inputs[[id]]
      output_name = private$.outputs[[id]]
      # every input channel of the current layer is the dst_channel exactly once
      # Not really sure why we sort according to the name afterwards
      input_tbl = edges[get("dst_id") == id, list(name = get("dst_channel"), payload = get("payload"))][input_name, , on = "name"]
      # we clear the input of the current layer
      edges[get("dst_id") == id, "payload" := list(list(NULL))]
      # and generate the input for the current layer
      input = input_tbl$payload
      names(input) = input_tbl$name

      lg$debug("Running PipeOp '%s$%s()'", id, fun, pipeop = op, input = input)

      # this now either outputs a tensor or a named list of tensors
      output = do.call(layer, args = input)
      if (is.null(names(output))) {
        if (inherits(output, "torch_tensor")) {
          output = set_names(list(output), output_name)
        } else { # is already a list
          output = set_names(output, output_name)
        }
      }

      edges[list(id, output_name), "payload" := list(..output[get("src_channel")]), on = c("src_id", "src_channel")]
    }
    return(output[[1L]])
  },
  add_edge = function(src_id, dst_id, src_channel, dst_channel) {
    row = data.table(src_id, src_channel, dst_id, dst_channel)
    private$.edges = rbind(private$.edges, row, fill = TRUE)
  },
  add_layer = function(id, layer, inputs, outputs) {
    assert_true(id %nin% private$.ids)
    self[[id]] = layer
    private$.ids = c(private$.ids, id)
    private$.inputs[[id]] = inputs
    private$.outputs[[id]] = outputs
  },
  active = list(
    edges = function(rhs) {
      assert_ro_binding(rhs)
      private$.edges
    }
  )
)
