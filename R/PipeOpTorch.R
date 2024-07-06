#' @title Base Class for Torch Module Constructor Wrappers
#'
#' @name mlr_pipeops_torch
#'
#' @description
#' `PipeOpTorch` is the base class for all [`PipeOp`][mlr3pipelines::PipeOp]s that represent
#' neural network layers in a [`Graph`][mlr3pipelines::Graph].
#' During **training**, it generates a [`PipeOpModule`] that wraps an [`nn_module`][torch::nn_module] and attaches it
#' to the architecture, which is also represented as a [`Graph`][mlr3pipelines::Graph] consisting mostly of [`PipeOpModule`]s
#' an [`PipeOpNOP`][mlr3pipelines::PipeOpNOP]s.
#'
#' While the former [`Graph`][mlr3pipelines::Graph] operates on [`ModelDescriptor`]s, the latter operates on [tensors][torch::torch_tensor].
#'
#' The relationship between a `PipeOpTorch` and a [`PipeOpModule`] is similar to the
#' relationshop between a `nn_module_generator` (like [`nn_linear`][torch::nn_linear]) and a
#' [`nn_module`][torch::nn_module] (like the output of `nn_linear(...)`).
#' A crucial difference is that the `PipeOpTorch` infers auxiliary parameters (like `in_features` for
#' `nn_linear`) automatically from the intermediate tensor shapes that are being communicated through the
#' [`ModelDescriptor`].
#'
#' During **prediction**, `PipeOpTorch` takes in a [`Task`][mlr3::Task] in each channel and outputs the same new
#' [`Task`][mlr3::Task] resulting from their [feature union][mlr3pipelines::PipeOpFeatureUnion] in each channel.
#' If there is only one input and output channel, the task is simply piped through.
#'
#' @section Inheriting:
#' When inheriting from this class, one should overload either the `private$.shapes_out()` and the
#' `private$.shape_dependent_params()` methods, or overload `private$.make_module()`.
#'
#' * `.make_module(shapes_in, param_vals, task)`\cr
#'   (`list()`, `list()`) -> `nn_module`\cr
#'   This private method is called to generated the `nn_module` that is passed as argument `module` to
#'   [`PipeOpModule`]. It must be overwritten, when no `module_generator` is provided.
#'   If left as is, it calls the provided `module_generator` with the arguments obtained by
#'   the private method `.shape_dependent_params()`.
#' * `.shapes_out(shapes_in, param_vals, task)`\cr
#'   (`list()`, `list()`, [`Task`][mlr3::Task] or `NULL`) -> named `list()`\cr
#'   This private method gets a list of `numeric` vectors (`shapes_in`), the parameter values (`param_vals`),
#'   as well as an (optional) [`Task`][mlr3::Task].
#    The `shapes_in` list indicates the shape of input tensors that will be fed to the module's `$forward()` function.
#    The list has one item per input tensor, typically only one.
#    The function should return a list of shapes of tensors that are created by the module.
#'   The `shapes_in` can be assumed to be in the same order as the input names of the `PipeOp`.
#'   The output shapes must be in the same order as the output names of the `PipeOp`.
#'   In case the output shapes depends on the task (as is the case for [`PipeOpTorchHead`]), the function should return
#'   valid output shapes (possibly containing `NA`s) if the `task` argument is provided or not.
#' * `.shape_dependent_params(shapes_in, param_vals, task)`\cr
#'   (`list()`, `list()`) -> named `list()`\cr
#'   This private method has the same inputs as `.shapes_out`.
#'   If `.make_module()` is not overwritten, it constructs the arguments passed to `module_generator`.
#'   Usually this means that it must infer the auxiliary parameters that can be inferred from the input shapes
#'   and add it to the user-supplied parameter values (`param_vals`).
#'
#' @section Input and Output Channels:
#' During *training*, all inputs and outputs are of class [`ModelDescriptor`].
#' During *prediction*, all input and output channels are of class [`Task`][mlr3::Task].
#'
#' @template pipeop_torch_state_default
#'
#' @section Parameters:
#' The [`ParamSet`][paradox::ParamSet] is specified by the child class inheriting from [`PipeOpTorch`].
#' Usually the parameters are the arguments of the wrapped [`nn_module`][torch::nn_module] minus the auxiliary parameter that can
#' be automatically inferred from the shapes of the input tensors.
#'
#' @section Internals:
#' During training, the `PipeOpTorch` creates a [`PipeOpModule`] for the given parameter specification and the
#' input shapes from the incoming [`ModelDescriptor`]s using the private method `.make_module()`.
#' The input shapes are provided by the slot `pointer_shape` of the incoming [`ModelDescriptor`]s.
#' The channel names of this [`PipeOpModule`] are identical to the channel names of the generating [`PipeOpTorch`].
#'
#' A [model descriptor union][model_descriptor_union] of all incoming [`ModelDescriptor`]s is then created.
#' Note that this modifies the [`graph`][mlr3pipelines::Graph] of the first [`ModelDescriptor`] **in place** for efficiency.
#' The [`PipeOpModule`] is added to the [`graph`][mlr3pipelines::Graph] slot of this union and the the edges that connect the
#' sending `PipeOpModule`s to the input channel of this `PipeOpModule` are addeded to the graph.
#' This is possible because every incoming [`ModelDescriptor`] contains the information about the
#' `id` and the `channel` name of the sending `PipeOp` in the slot `pointer`.
#'
#' The new graph in the [`model_descriptor_union`] represents the current state of the neural network
#' architecture. It is structurally similar to the subgraph that consists of all pipeops of class `PipeOpTorch` and
#' [`PipeOpTorchIngress`] that are ancestors of this `PipeOpTorch`.
#'
#' For the output, a shallow copy of the [`ModelDescriptor`] is created and the `pointer` and
#' `pointer_shape` are updated accordingly. The shallow copy means that all [`ModelDescriptor`]s point to the same
#' [`Graph`][mlr3pipelines::Graph] which allows the graph to be modified by-reference in different parts of the code.
#' @export
#' @family Graph Network
#' @examplesIf torch::torch_is_installed()
#' ## Creating a neural network
#' # In torch
#'
#' task = tsk("iris")
#'
#' network_generator = torch::nn_module(
#'   initialize = function(task, d_hidden) {
#'     d_in = length(task$feature_names)
#'     self$linear = torch::nn_linear(d_in, d_hidden)
#'     self$output = if (task$task_type == "regr") {
#'       torch::nn_linear(d_hidden, 1)
#'     } else if (task$task_type == "classif") {
#'       torch::nn_linear(d_hidden, length(task$class_names))
#'     }
#'   },
#'   forward = function(x) {
#'     x = self$linear(x)
#'     x = torch::nnf_relu(x)
#'     self$output(x)
#'   }
#' )
#'
#' network = network_generator(task, d_hidden = 50)
#' x = torch::torch_tensor(as.matrix(task$data(1, task$feature_names)))
#' y = torch::with_no_grad(network(x))
#'
#'
#' # In mlr3torch
#' network_generator = po("torch_ingress_num") %>>%
#'   po("nn_linear", out_features = 50) %>>%
#'   po("nn_head")
#' md = network_generator$train(task)[[1L]]
#' network = model_descriptor_to_module(md)
#' y = torch::with_no_grad(network(torch_ingress_num.input = x))
#'
#'
#'
#' ## Implementing a custom PipeOpTorch
#'
#' # defining a custom module
#' nn_custom = nn_module("nn_custom",
#'   initialize = function(d_in1, d_in2, d_out1, d_out2, bias = TRUE) {
#'     self$linear1 = nn_linear(d_in1, d_out1, bias)
#'     self$linear2 = nn_linear(d_in2, d_out2, bias)
#'   },
#'   forward = function(input1, input2) {
#'     output1 = self$linear1(input1)
#'     output2 = self$linear1(input2)
#'
#'     list(output1 = output1, output2 = output2)
#'   }
#' )
#'
#' # wrapping the module into a custom PipeOpTorch
#'
#' library(paradox)
#'
#' PipeOpTorchCustom = R6::R6Class("PipeOpTorchCustom",
#'   inherit = PipeOpTorch,
#'   public = list(
#'     initialize = function(id = "nn_custom", param_vals = list()) {
#'       param_set = ps(
#'         d_out1 = p_int(lower = 1, tags = c("required", "train")),
#'         d_out2 = p_int(lower = 1, tags = c("required", "train")),
#'         bias = p_lgl(default = TRUE, tags = "train")
#'       )
#'       super$initialize(
#'         id = id,
#'         param_vals = param_vals,
#'         param_set = param_set,
#'         inname = c("input1", "input2"),
#'         outname = c("output1", "output2"),
#'         module_generator = nn_custom
#'       )
#'     }
#'   ),
#'   private = list(
#'     .shape_dependent_params = function(shapes_in, param_vals, task) {
#'       c(param_vals,
#'         list(d_in1 = tail(shapes_in[["input1"]], 1)), d_in2 = tail(shapes_in[["input2"]], 1)
#'       )
#'     },
#'     .shapes_out = function(shapes_in, param_vals, task) {
#'       list(
#'         input1 = c(head(shapes_in[["input1"]], -1), param_vals$d_out1),
#'         input2 = c(head(shapes_in[["input2"]], -1), param_vals$d_out2)
#'       )
#'     }
#'   )
#' )
#'
#' ## Training
#'
#' # generate input
#' task = tsk("iris")
#' task1 = task$clone()$select(paste0("Sepal.", c("Length", "Width")))
#' task2 = task$clone()$select(paste0("Petal.", c("Length", "Width")))
#' graph = gunion(list(po("torch_ingress_num_1"), po("torch_ingress_num_2")))
#' mds_in = graph$train(list(task1, task2), single_input = FALSE)
#'
#' mds_in[[1L]][c("graph", "task", "ingress", "pointer", "pointer_shape")]
#' mds_in[[2L]][c("graph", "task", "ingress", "pointer", "pointer_shape")]
#'
#' # creating the PipeOpTorch and training it
#' po_torch = PipeOpTorchCustom$new()
#' po_torch$param_set$values = list(d_out1 = 10, d_out2 = 20)
#' train_input = list(input1 = mds_in[[1L]], input2 = mds_in[[2L]])
#' mds_out = do.call(po_torch$train, args = list(input = train_input))
#' po_torch$state
#'
#' # the new model descriptors
#'
#' # the resulting graphs are identical
#' identical(mds_out[[1L]]$graph, mds_out[[2L]]$graph)
#' # not that as a side-effect, also one of the input graphs is modified in-place for efficiency
#' mds_in[[1L]]$graph$edges
#'
#' # The new task has both Sepal and Petal features
#' identical(mds_out[[1L]]$task, mds_out[[2L]]$task)
#' mds_out[[2L]]$task
#'
#' # The new ingress slot contains all ingressors
#' identical(mds_out[[1L]]$ingress, mds_out[[2L]]$ingress)
#' mds_out[[1L]]$ingress
#'
#' # The pointer and pointer_shape slots are different
#' mds_out[[1L]]$pointer
#' mds_out[[2L]]$pointer
#'
#' mds_out[[1L]]$pointer_shape
#' mds_out[[2L]]$pointer_shape
#'
#' ## Prediction
#' predict_input = list(input1 = task1, input2 = task2)
#' tasks_out = do.call(po_torch$predict, args = list(input = predict_input))
#' identical(tasks_out[[1L]], tasks_out[[2L]])
PipeOpTorch = R6Class("PipeOpTorch",
  inherit = PipeOp,
  cloneable = FALSE,
  public = list(
    #' @field module_generator (`nn_module_generator` or `NULL`)\cr
    #'    The module generator wrapped by this `PipeOpTorch`. If `NULL`, the private method
    #'    `private$.make_module(shapes_in, param_vals)` must be overwritte, see section 'Inheriting'.
    #'    Do not change this after construction.
    module_generator = NULL,
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @template param_module_generator
    #' @template param_param_set
    #' @template param_packages
    #' @param tags (`character()`)\cr
    #'   The tags of the [`PipeOp`][mlr3pipelines::PipeOp]. The tags `"torch"` is always added.
    #' @param inname (`character()`)\cr
    #'   The names of the [`PipeOp`][mlr3pipelines::PipeOp]'s input channels. These will be the input channels of the generated [`PipeOpModule`].
    #'   Unless the wrapped `module_generator`'s forward method (if present) has the argument `...`, `inname` must be
    #'   identical to those argument names in order to avoid any ambiguity.\cr
    #'   If the forward method has the argument `...`, the order of the input channels determines how the tensors
    #'   will be passed to the wrapped `nn_module`.\cr
    #'   If left as `NULL` (default), the argument `module_generator` must be given and the argument names of the
    #'   `modue_generator`'s forward function are set as `inname`.
    #' @param outname (`character()`) \cr
    #'   The names of the output channels channels. These will be the ouput channels of the generated [`PipeOpModule`]
    #'   and therefore also the names of the list returned by its `$train()`.
    #'   In case there is more than one output channel, the `nn_module` that is constructed by this
    #'   [`PipeOp`][mlr3pipelines::PipeOp] during training must return a named `list()`, where the names of the list are the
    #'   names out the output channels. The default is `"output"`.
    initialize = function(id, module_generator, param_set = ps(), param_vals = list(),
      inname = "input", outname = "output", packages = "torch", tags = NULL) {
      self$module_generator = assert_class(module_generator, "nn_module_generator", null.ok = TRUE)
      assert_character(inname, .var.name = "input channel names")
      assert_character(outname, .var.name = "output channel names", min.len = 1L)
      assert_character(tags, null.ok = TRUE)
      assert_character(packages, any.missing = FALSE)

      packages = union(packages, c("mlr3torch", "torch"))
      input = data.table(name = inname, train = "ModelDescriptor", predict = "Task")
      output = data.table(name = outname, train = "ModelDescriptor", predict = "Task")
      private$.only_shape = FALSE

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output,
        packages = packages,
        tags = unique(c("torch", tags))
      )
    },
    #' @description
    #'  Calculates the output shapes for the given input shapes, parameters and task.
    #' @param shapes_in (`list()` of `integer()`)\cr
    #'   The input input shapes, which must be in the same order as the input channel names of the `PipeOp`.
    #' @param task ([`Task`][mlr3::Task] or `NULL`)\cr
    #'  The task, which is very rarely used (default is `NULL`). An exception is [`PipeOpTorchHead`].
    #' @return
    #'  A named `list()` containing the output shapes. The names are the names of the output channels of
    #'  the `PipeOp`.
    shapes_out = function(shapes_in, task = NULL) {
      assert_r6(task, "Task", null.ok = TRUE)
      if (is.numeric(shapes_in)) shapes_in = list(shapes_in)
      # batch dimension can be known or unknown
      assert_shapes(shapes_in, unknown_batch = NULL)
      if ("..." %nin% self$input$name) {
        assert_true(length(shapes_in) == nrow(self$input),
          .var.name = "number of input shapes equal to number of input channels")
      }
      set_names(private$.shapes_out(shapes_in, self$param_set$get_values(), task = task), self$output$name)
    }

    # TODO: printer that calls the nn_module's printer
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) shapes_in,
    .shape_dependent_params = function(shapes_in, param_vals, task) param_vals,
    .make_module = function(shapes_in, param_vals, task) {
      do.call(self$module_generator, private$.shape_dependent_params(shapes_in, param_vals, task))
    },
    .train = function(inputs) {
      param_vals = self$param_set$get_values()
      input_pointers = map(inputs, "pointer")
      input_shapes = map(inputs, "pointer_shape")

      # Now begin creating the result-object: it contains a merged version of all `inputs`' $graph slots etc.
      # The only thing missing afterwards is (1) integrating module_op to the merged $graph, and adding `pointer`s.
      result_template = Reduce(model_descriptor_union, inputs)
      task = result_template$task

      # first user-supplied function: infer shapes that get created for module
      shapes_out = self$shapes_out(input_shapes, task)
      shapes_out = assert_list(shapes_out, types = "numeric", any.missing = FALSE, len = nrow(self$output))

      # we need this so PipeOpBlock can implement $shapes_out() without creating the possibly expensive network
      if (!private$.only_shape) {
        # second possibly user-supplied function: create the concrete nn_module, given shape info.
        # If this is not user-supplied, then at least `.shape_dependent_params` is called.
        module = private$.make_module(input_shapes, param_vals, task)

        # create the PipeOp that contains the instantiated nn_module.
        module_op = PipeOpModule$new(
          id = self$id,
          module = module,
          inname = self$input$name,
          outname = self$output$name,
          packages = self$packages
        )

        # integrate the operation into the graph
        result_template$graph$add_pipeop(module_op, clone = FALSE)
        # All of the `inputs` contained possibly the same `graph`, but definitely had different `pointer`s,
        # indicating the different channels from within the `graph` that should be connected to the new operation.
        vararg = "..." == module_op$input$name[[1L]]
        current_channel = "..."

        for (i in seq_along(inputs)) {
          ptr = input_pointers[[i]]
          if (!vararg) current_channel = module_op$input$name[[i]]
          result_template$graph$add_edge(
            src_id = ptr[[1]], src_channel = ptr[[2]],
            dst_id = module_op$id, dst_channel = current_channel
          )
        }
      }

      # now we split up the result_template into one item per output channel.
      # each output channel contains a different `pointer` / `pointer_shape`, referring to the
      # individual outputs of the module_op.
      results = Map(shape = shapes_out, channel_id = self$output$name, f = function(shape, channel_id) {
        r = result_template  # unnecessary, but good for readability: result_template is not changed
        r$pointer = c(self$id, channel_id)
        r$pointer_shape = shape
        r
      })
      # PipeOp API requires us to only set this to some list. We set it to output shape to ease debugging.
      self$state = shapes_out

      results
    },
    .predict = function(inputs) {
      # here we just pipe the Tasks through
      if (length(inputs) > 1) {
        inputs = PipeOpFeatureUnion$new()$train(inputs)
      }
      rep(inputs[1], nrow(self$output))
    },
    .only_shape = NULL
  )
)
