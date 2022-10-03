#' @title Base Class for Torch Module Constructor Wrappers
#'
#' @usage NULL
#' @name mlr_pipeops_torch
#' @format [`R6Class`] object inheriting from [`PipeOpTaskPreproc`]/[`PipeOp`].
#'
#' @description
#' `PipeOpTorch` wraps a `torch::nn_module_generator`. It builds an isomorphic `Graph` and also keeps track of the
#' tensor shape(s) involved.
#'
#' @section Input and Output Channels:
#'
#' During training, all input channels take in a [ModelDescriptor] and output a [ModelDescriptor].
#' During prediction, all input channels take in a [mlr3::Task] and output the same [mlr3::Task].
#'
#' The generated [PipeOpModule] has exactly the same input and output channels.
#' The names of the input channels correspond to the argument names of the generated `nn_module`.
#' The output channels should correspond to the
#'
#'
#'
#' The output is the input [`Task`][mlr3::Task] with all affected numeric features replaced by their
#' non-negative components.
#'
#' @section Inheriting:
#' When inheriting from this class, one should overload the `private$.shapes_out()` and the `private$.shape_dependent_
#' params()` methods.
#'
#' `private$.shapes_out()` gets a list of `numeric` vectors as input. This list indicates the shape of input tensors
#' that will be fed to the module's `$forward()` function. The list has one item per input tensor, typically only one.
#' The shape vectors may contain `NA` entries indicating arbitrary size (typically used for the batch dimension). The
#' function should return a list of shapes of tensors that are created by the module. If `multi_output` is `NULL` it
#' should be of length 1. If `multi_output` is an `integer(1)`, then this should be the sgraph = ame length as the return value
#' of the module's `$forward()`.
#'
#' `private$.shape_dependent_params()` calculates construction arguments of `module_generator` that depend on the input shape.
#' It should return a named list.
#'
#' It is possible to overload `private$.make_module()` instead of `private$.shape_dependent_params()`, in which case it is even allowed
#' to not give a `module_generator` during construction.
#'
#' @param module_generator (`nn_module_generator` | `NULL`)\cr
#'   When this is `NULL`, then `private$.make_module` must be overloaded.
#' @param inname (`character()` or `NULL`)\cr
#'   The names of the [PipeOp]'s input channels.
#'   If `module_generator` is not `NULL`, they must be a permutation of the argument names of the module's `forward`
#'   function. This is also the default behaviour
#'   inferred from the forward function of the module generator.
#'   Otherwise this can be a character vector giving the names, or an integer `n`, such that the
#'   names are `input1, input2, ..., input<n>`.
#' @param outname (`character()`) \cr
#'   The names of the [PipeOp]'s output channels channels.
#'   The default is `"output"`.
#'   In case there is more than one output channel, the `nn_module` that is constructed by this
#'   [PipeOp] during training must return a named list, where the names of the list are the
#'   names out the output channels. When providing an integer `n`, the names are set to
#'   `input1, input2, ..., input<n>`.
#' @template param_id
#' @template param_param_set
#' @template param_param_vals
#' @param packages (`character()`)\cr
#'    The R packages this [PipeOP] depends on.
#' @examples
#' @export
PipeOpTorch = R6Class("PipeOpTorch",
  inherit = PipeOp,
  public = list(
    module_generator = NULL,
    #' @description Initializes a new instance of this [R6 Class][R6::R6Class].
    initialize = function(id, module_generator, param_set = ps(), param_vals = list(),
      inname = NULL, outname = "output", packages = "torch") {
      self$module_generator = assert_class(module_generator, "nn_module_generator", null.ok = TRUE)
      assert_names(outname, type = "strict")
      assert(check_names(inname, type = "strict"), if (is.null(module_generator)) TRUE else check_true(is.null(inname)))

      if (!is.null(module_generator)) {
        argnames = formalArgs(module_generator$public_methods$forward)
        if (is.null(inname)) {
          inname = argnames
        } else {
          assert_true(all(sort(inname) == sort(argnames)))
        }
      }

      assert_character(packages, any.missing = FALSE)
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
        inname = self$input$name,
        outname = self$output$name,
        packages = self$packages
      )

      # Now begin creating the result-object: it contains a merged version of all `inputs`' $graph slots etc.
      # The only thing missing afterwards is (1) integrating module_op to the merged $graph, and adding `.pointer`s.
      result_template = Reduce(model_descriptor_union, inputs)

      # integrate the operation into the graph
      result_template$graph$add_pipeop(module_op)
      # All of the `inputs` contained possibly the same `graph`, but definitely had different `.pointer`s,
      # indicating the different channels from within the `graph` that should be connected to the new operation.
      for (i in seq_along(inputs)) {
        ptr = input_pointers[[i]]
        current_channel = module_op$input$name[[i]]
        result_template$graph$add_edge(
          src_id = ptr[[1]], src_channel = ptr[[2]],
          dst_id = module_op$id, dst_channel = current_channel
        )
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
      rep(inputs[1], nrow(self$output))
    }
  )
)


