#' @title Base Class for Torch Module Constructor Wrappers
#'
#' @description
#' `PipeOpTorch` wraps a `torch::nn_module_generator`. It builds an isomorphic `Graph` and also keeps track of the tensor shape(s) involved.
#'
#' @section Inheriting:
#' When inheriting from this class, one should overload the `private$.shapes_out()` and the `private$.shape_dependent_params()`
#' methods.
#'
#' `private$.shapes_out()` gets a list of `numeric` vectors as input. This list indicates the shape of input tensors that will be fed
#' to the module's `$forward()` function. The list has one item per input tensor, typically only one. The shape vectors may contain `NA`
#' entries indicating arbitrary size (typically used for the batch dimension). The function should return a list of shapes of tensors that
#' are created by the module. If `multi_output` is `NULL` it should be of length 1. If `multi_output` is an `integer(1)`, then
#' this should be the same length as the return value of the module's `$forward()`.
#'
#' `private$.shape_dependent_params()` calculates construction arguments of `module_generator` that depend on the input shape.
#' It should return a named list.
#'
#' It is possible to overload `private$.make_module()` instead of `private$.shape_dependent_params()`, in which case it is even allowed
#' to not give a `module_generator` during construction.
#'
#' @examples
#' @export
PipeOpTorch = R6Class("PipeOpTorch",
  inherit = PipeOp,
  public = list(
    module_generator = NULL,
    #' @description Initializes a new instance of this [R6 Class][R6::R6Class].
    #' @param module_generator (`nn_module_generator` | `NULL`)\cr
    #'   When this is `NULL`, then `private$.make_module` must be overloaded.
    #' @param multi_input (`NULL` | `integer(1)`)\cr
    #'   `0`: `...`-input. Otherwise: `multi_input` times input channel named `input1:`...`input#`.\cr
    #'   `module`'s `$forward` function must take `...`-input if `multi_input` is 0, and must have `multi_input` arguments otherwise.
    #' @param multi_output (`NULL` | `integer(1)`)\cr
    #'   `NULL`: single output. Otherwise: `multi_output` times output channel named `output1:`...`input#`.\cr
    #'   `module`'s `$forward` function must return a `list` of `torch_tensor` if `multi_output` is not `NULL`.
    initialize = function(id, module_generator, param_set = ps(), param_vals = list(), multi_input = 1, multi_output = NULL, packages = character(0)) {
      # default input and output channels, packages
      assert_int(multi_input, null.ok = TRUE, lower = 0)
      assert_int(multi_output, null.ok = TRUE, lower = 1)
      self$module_generator = assert_class(module, "nn_module_generator", null.ok = TRUE)
      assert_character(packages, any.missing = FALSE)

      inname = if (multi_input == 0) "..." else sprintf("input%s", seq_len(multi_input))
      outname = if (is.null(multi_output)) "output" else sprintf("output%s", seq_len(multi_output))
      input = data.table(name = inname, train = "ModelDescriptor", predict = "Task")
      output = data.table(name = outname, train = "ModelDescriptor", predict = "Task")

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output,
        packages = packages
      )
    },
    shapes_out = function(shapes_in) {
      if (is.numeric(shapes_in)) shapes_in = list(shapes_in)
      if (identical(self$input$name, "...")) {
        assert_list(shapes_in, min.len = 1, types = "numeric")
      } else {
        assert_list(shapes_in, len = nrow(self$input), types = "numeric")
      }
      pv = self$param_set$get_values()
      (assert_list(private$.shapes_out(shapes_in, pv), len = nrow(self$output), types = "numeric"))
    }
    # TODO: printer that calls the nn_module's printer
    # TODO: maybe call input just 'input' and not 'input1' if only one input present
    # TODO: make module a read-only active binding
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals) shapes_in,
    .shape_dependent_params = function(shapes_in, param_vals) param_vals,
    .make_module = function(shapes_in, param_vals) {
      do.call(self$module_generator, private$.shape_dependent_params(shapes_in, param_vals))
    },
    .train = function(inputs) {
      param_vals = self$param_set$get_values()
      input_pointers = map(inputs, ".pointer")
      input_shapes = map(inputs, ".pointer_shape")

      # first user-supplied function: infer shapes that get created for modul
      shapes_out = private$.shapes_out(input_shapes, param_vals)
      shapes_out = assert_list(shapes_out, types = "numeric", any.missing = FALSE, len = nrow(self$output))

      # second possibly user-supplied function: create the concrete nn_module, given shape info.
      # If this is not user-supplied, then at least `.shape_dependent_params` is called.
      module = private$.make_module(input_shapes, param_vals)

      # create the PipeOp that contains the instantiated nn_module.
      module_op = PipeOpModule$new(
        id = self$id,
        module = module,
        multi_input = nrow(self$input),
        multi_output = if (!identical(self$output$name, "output")) nrow(self$output),
        packages = self$packages
      )

      # Now begin creating the result-object: it contains a merged version of all `inputs`' $graph slots etc.
      # The only thing missing afterwards is (1) integrating module_op to the merged $graph, and adding `.pointer`s.
      result_template = Reduce(inputs, model_descriptor_union)

      # integrate the operation into the graph
      result_template$graph$add_pipeop(module_op)
      # All of the `inputs` contained possibly the same `graph`, but definitely had different `.pointer`s,
      # indicating the different channels from within the `graph` that should be connected to the new operation.
      # However, a `.pointer` may also refer directly to an input-shape -- especially when the `graph` is initially
      # empty -- in which case we leave the respective input of the new module_op empty and instead update the input_map.
      for (i in seq_along(inputs)) {
        ptr = input_pointers[[i]]
        current_channel = module_op$input$name[[i]]
        if (length(ptr == 1)) {  # pointer refers to input shape
          # global_inchannel is the entry of graph$input$name that refers to module_op's i'th input.
          global_inchannel = sprintf("%s.%s", module_op$id, current_channel)
          assert_true(global_inchannel %nin% names(result_template$input_map))
          result_template$input_map[[global_inchannel]] = ptr
        } else {
          result_template$graph$add_edge(
            src_id = ptr[[1]], src_channel = ptr[[2]],
            dst_id = module_op$id, dst_channel = current_channel
          )
        }
      }

      # now we split up the result_template into one item per output channel.
      # each output channel contains a different `.pointer` / `.pointer_shape`, referring to the
      # individual outputs of the module_op.
      results = Map(shape = shapes_out, channel_id = module_op$output$name, f = function(shape, channel_id) {
        r = result_template  # unnecessary, but good for readability: result_template is not changed
        r$.pointer = c(module_op$id, channel_id)
        r$.pointer_shape = shape
        r
      })

      self$state = shapes_out  # PipeOp API requires us to only set this to some list. We set it to output shape to ease debugging.

      results
    },
    .predict = function(inputs) {
      # here we just pipe the Tasks through
      if (length(inputs) > 1) {
        inputs = PipeOpFeatureUnion$new()$train(inputs)
      }
      rep(inputs[[1]], nrow(self$output))
    }
  )
)


