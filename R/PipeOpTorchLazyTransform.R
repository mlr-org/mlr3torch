#' @title Base Class for Lazy Transformations
#' @name mlr_pipeops_torch_lazy_transform
#'
#' @description
#' This is the base class for lazy tensor transformations.
#' These can either be used as preprocessing for [`lazy_tensor`] objects or as part of the
#' graph contained in the [`ModelDescriptor`].
#' Currently the `PipeOp` must have exactly one inut and one output.
#'
#' @section Inheriting:
#' You need to:
#' * Either provide a function as argument `fn` or overwrite the private `.make_module()` method.
#'   See [`PipeOpTorch`] for how to do this.
#' * In case the transformation changes the tensor shapes you must provide a private `.shapes_out()` method like
#'   for [`PipeOpTorch`]. It can be assumed that the first dimension is `NA`, i.e. the batch dimension.
#'
#' Depending on the data-loader, this function is either applied individually to each tensor from a batch and then
#' concatenated, or to the whole batch at once and should return the same result in both cases.
#'
#' @section Input and Output Channels:
#' During *training*, all inputs and outputs are of class [`Task`] or [`ModelDescriptor`].
#' During *prediction*, all inputs and outputs are of class [`Task`] or [`ModelDescriptor`].
#'
#' @template pipeop_torch_state_default
#' @section Parameters:
#' The [`ParamSet`][paradox::ParamSet] is specified by the child class inheriting from [`PipeOpTorchLazyTransform`].
#' The name `augment` is reserved and must not be used.
#'
#' @template param_id
#' @template param_param_vals
#' @template param_param_set
#' @param packages (`character()`)\cr
#'   The packages the function depends on.
#'   If `fn` is from a namespace and no value is provided the namespace is automatically set.
#' @param fn (`function()`)\cr
#'   A function that will be applied to a (lazy) tensor.
#'   Additional arguments can be passed as parameters.
#'   During actual preprocessing the (lazy) tensor will be passed by position.
#'
#' @export
PipeOpTorchLazyTransform = R6Class("PipeOpTorchLazyTransform",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(fn, id = "lazy_transform", param_vals = list(), param_set = ps(), packages = character(0)) {
      private$.fn = assert_function(fn, null.ok = TRUE)
      if (isNamespace(environment(fn)) && (length(packages) == 0)) {
        packages = getNamespaceName(environment(fn))
      }

      param_set$add(ps(
        augment = p_lgl(tags = c("train", "predict"))
      ))
      param_set$set_values(augment = FALSE)

      super$initialize(
        id = id,
        # TODO: Generalize this to arbitrary number of inputs and outputs
        inname = "input",
        outname = "output",
        param_vals = param_vals,
        param_set = param_set,
        packages = packages,
        module_generator = NULL,
        variant_types = TRUE
      )

    }
  ),
  private = list(
    .fn = NULL,
    .make_module = function(shapes_in, param_vals, task) {
      augment = param_vals$augment
      param_vals$augment = NULL
      trafo = private$.fn
      fn = crate(function(x) {
        invoke(.f = trafo, x, .args = param_vals)
      }, param_vals, trafo, .parent = topenv())

      if (augment) {
        nn_module(self$id,
          initialize = function(fn) {
            self$fn = fn
          },
          forward = function(x) {
            if (self$training) {
              self$fn(x)
            } else {
              x
            }
          }
        )(fn)
      } else {
        nn_module(self$id,
          initialize = function(fn) {
            self$fn = fn
          },
          forward = function(x) {
            self$fn(x)
          }
        )(fn)
      }
    },
    .train = function(inputs) {
      if (test_class(inputs[[1L]], "ModelDescriptor")) {
        super$.train(inputs)
      } else if (test_class(inputs[[1L]], "Task")) {
        private$.transform_task(inputs)
      } else {
        stopf("Unsupported input type '%s'.", class(inputs)[[1L]])
      }
    },
    .predict = function(inputs) {
      if (test_class(inputs[[1L]], "ModelDescriptor")) {
        super$.predict(inputs)
      } else if (test_class(inputs[[1L]], "Task")) {
        private$.transform_task(inputs)
      }
    },
    .transform_task = function(inputs) {
      intask = inputs[[1L]]$clone(deep = TRUE)

      lazy_cols = intask$feature_types[get("type") == "lazy_tensor", "id"][[1L]]
      if (length(lazy_cols) != 1L) {
        stopf("Can only use PipeOpTorchLazyTransform on tasks with exactly one lazy tensor column.")
      }
      lt = intask$data(cols = lazy_cols)[[1L]]

      self$state = list(lazy_cols)

      shapes_in = attr(lt[[1L]], "data_descriptor")$.pointer_shape

      fn = private$.make_module(shapes_in = shapes_in, param_vals = self$param_set$values, task = intask)

      po_fn = PipeOpModule$new(
        id = self$id,
        module = fn,
        inname = self$input$name,
        outname = self$output$name,
        packages = self$packages
      )

      shapes_out = self$shapes_out(shapes_in, intask)
      lt_processed = transform_lazy_tensor(lt, po_fn, shapes_out[[1L]])
      new_col = set_names(data.table(lt_processed), lazy_cols)

      list(intask$select(setdiff(intask$feature_names, lazy_cols))$cbind(new_col))
    },
    .additional_phash_input = function() {
      list(self$param_set$ids(), private$.fn, self$packages)
    }
  )
)

#' @include zzz.R
register_po("lazy_transform", PipeOpTorchLazyTransform)


#' @title Resize an Object
#' @description
#' See [`torchvision::transform_resize()`] for more information.
#' Creates a new instance of this [R6][R6::R6Class] class.
#' @template param_id
#' @template param_param_vals
#' @export
PipeOpTorchTransformResize = R6Class("PipeOpTOrchTransformResize",
  inherit = PipeOpTorchLazyTransform,
  public = list(
    initialize = function(id = "transform_resize", param_vals = list()) {
      param_set = ps(
        size = p_uty(tags = c("train", "required")),
        interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
          tags = "train", default = 2L
        )
      )
      super$initialize(
        id = id,
        packages = "torchvision",
        param_set = param_set,
        param_vals = param_vals,
        fn = torchvision::transform_resize
      )
    }
  ),
  private = list(
    .shapes_out = function(shapes_in, param_vals, task) {
      size = param_vals$size
      shape = shapes_in[[1L]]
      assert_true(length(shape) > 2)
      height = shape[[length(shape) - 1L]]
      width = shape[[length(shape)]]
      s = torchvision::transform_resize(torch_ones(c(1, height, width), device = "meta"), size = size)$shape[2:3]
      list(c(shape[seq_len(length(shape) - 2L)], s))
    }
  )
)

#' @include zzz.R
register_po("transform_resize", PipeOpTorchTransformResize)
