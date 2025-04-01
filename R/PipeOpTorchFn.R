#' @title Custom Function
#' @name mlr_pipeops_nn_fn
#' 
#' @description
#' Applies a user-supplied function to a tensor.

#' @section Parameters:
#' * `fn` :: `function`\cr
#'   The function to apply. Takes a `torch` tensor as its first argument and returns a `torch` tensor.
#'
#' @templateVar id nn_fn
#' @template pipeop_torch_channels_default
#'
#' @example
#' 
#' @export
PipeOpTorchFn = R6Class("PipeOpTorchFn",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_fn", param_vals = list()) {
      param_set = ps(
        fn = p_uty(tags = c("train", "required"), custom_check = check_function),
        shapes_out = p_uty(tags = "train", custom_check = check_function_or_null)
      )

      ps_fn = inferps(param_vals$fn)

      super$initialize(
        id = id,
        param_set = c(param_set, ps_fn),
        param_vals = param_vals,
        module_generator = NULL
      )
    }
  ),
  private = list(
    .shapes_out =function(shapes_in, param_vals, task) {
      sin = shapes_in[["input"]]
      batch_dim = sin[1L]
      batchdim_is_unknown = is.na(batch_dim)
      if (batchdim_is_unknown) {
        sin[1] = 1L
      }
      tensor_in = mlr3misc::invoke(torch_empty, .args = sin, device = torch_device("cpu"))
      tensor_out = tryCatch(mlr3misc::invoke(param_vals$fn, tensor_in, .args = param_vals),
        error = function(e) {
          stopf("Input shape '%s' is invalid for PipeOp with id '%s'.", shape_to_str(list(sin)), self$id)
        }
      )
      sout = dim(tensor_out)
      if (batchdim_is_unknown) {
        sout[1] = NA
      }

      list(sout)
    },
    .make_module = function(shapes_in, param_vals, task) {
      nn_module("nn_fn",
        initialize = function(fn) {
          self$fn = fn
        },
        forward = function(x) {
          return(self$fn(x))
        }
      )(param_vals$fn)
    }
  )
)

check_tensor_function = function(input) {
  is_function = check_function(input)
  if (!is_function == TRUE) {
    return(is_function)
  }
  
  args = names(formals(input))
  if (length(args) < 1) {
    return("Function must have at least one argument for tensor input")
  }
  
  tryCatch({
    test_tensor = torch::torch_empty(c(1, 2, 3), device = torch_device("meta"))
    result = input(test_tensor)
    
    if (!inherits(result, "torch_tensor")) {
      return("Function must return a torch_tensor")
    }

    return(TRUE)
  }, error = function(e) {
    return(sprintf("Error when testing function with tensor input: %s", e$message))
  })
}

check_shapes_out = function(input) {
  is_function = check_function(input, args = c("shapes_in", "param_vals", "task"))
  if (!is_function == TRUE) {
    return(is_function)
  }

  tryCatch({
    mock_shapes_in = list(input = c(NA, 10, 10))
    mock_param_vals = list(fn = function(x) x)
    mock_task = NULL
    result = input(mock_shapes_in, mock_param_vals, mock_task)
    
    is_list = check_list(result)
    if (!is_list == TRUE) {
      return(is_list)
    }
    
    for (i in seq_along(result)) {
      if (!is.numeric(result[[i]])) {
        return(sprintf("Item %d in result is not a numeric vector (shape)", i))
      }
      if (!is.na(result[[i]][1])) {
        return("First dimension of each shape must be NA (batch dimension)")
      }
    }

    return(TRUE)
  }, error = function(e) {
    return(sprintf("Error evaluating function: %s", e$message))
  })
}

#' @include aaa.R
register_po("nn_fn", PipeOpTorchFn)

      # begin Copilot
      # Get function from param_vals if provided, to infer its parameters

      # fn_args = names(formals(param_vals$fn))
    
      # Skip first argument (which should be the tensor input)
      # if (length(fn_args) > 1) {
      #   # Remove tensor input and any ... argument
      #   extra_args = setdiff(fn_args[-1], "...")
        
      #   if (length(extra_args) > 0) {
      #     # Get default values for the arguments
      #     fn_defaults = formals(param_vals$fn)[extra_args]
          
      #     # Add each argument to the parameter set
      #     for (arg in extra_args) {
      #       default_val = fn_defaults[[arg]]
      #       # If parameter has a default value, use it
      #       if (!identical(default_val, quote(expr=))) {
      #         # Evaluate default if needed (handles expressions like NA, NULL)
      #         if (is.call(default_val) || is.name(default_val)) {
      #           default_val = eval(default_val)
      #         }
      #         # Add parameter with default value
      #         param_set$add(ps(
      #           placeholder = p_uty(default = default_val, tags = "train")
      #         )$params[[1]]$set_id(arg))
      #       } else {
      #         # Add required parameter with no default
      #         param_set$add(ps(
      #           placeholder = p_uty(tags = c("train", "required"))
      #         )$params[[1]]$set_id(arg))
      #       }
      #     }
      #   }
      # }
      # end Copilot