# test_that("PipeOpTorchFn works for a simple function", {
#   withr::local_options(mlr3torch.cache = TRUE)
  
#   # for the nano imagenet data, gets the blue channel
#   drop_dim =  function(x, ...) x[, -1, , ]
#   po = po("nn_fn", param_vals = list(fn = drop_dim))
#   graph = po("torch_ingress_ltnsr") %>>% po

#   task = nano_imagenet()
#   task_dt = task$data()
  
#   graph$train(task)
#   result = graph$predict(task)[[1]]
#   result_dt = result$data()

#   result_dt
# })

test_that("PipeOpTorchFn works for a simple function", {
  withr::local_options(mlr3torch.cache = TRUE)
  
  # For debugging - first verify our function works directly on tensors
  test_tensor = torch_randn(c(2, 3, 8, 8))
  drop_dim = function(x, ...) {
    # Print shape for debugging
    cat("Input tensor shape:", paste(dim(x), collapse="x"), "\n")
    # Return modified tensor with first channel dropped
    result = x[, -1, , ]
    cat("Output tensor shape:", paste(dim(result), collapse="x"), "\n")
    return(result)
  }
  
  # Test function works directly
  test_result = drop_dim(test_tensor)
  expect_equal(dim(test_result)[2], dim(test_tensor)[2] - 1)
  
  # Now create the PipeOp
  po_relu = po("nn_relu")
  po = po("nn_fn", param_vals = list(fn = drop_dim))
  
  # Create graph with lazy tensor ingress
  graph = po("torch_ingress_ltnsr") %>>% po_relu
  graph = po("torch_ingress_ltnsr") %>>% po
  
  # Get nano_imagenet task
  task = nano_imagenet()
  task = tsk("lazy_iris")
  
  # Print original task structure
  cat("Task feature names:", task$feature_names, "\n")
  cat("Task target:", task$target_names, "\n")
  
  # IMPORTANT: Add data inspection
  task_data = task$data()
  if ("image" %in% names(task_data)) {
    # Print image tensor info if available
    sample_image = task_data$image[[1]]
    cat("Image sample shape:", paste(dim(sample_image), collapse="x"), "\n")
  }
  
  # Train the graph
  trained = graph$train(task)
  trained[[1]]$data()$x[1] |>
    materialize()

  result = graph$predict(task)
  result[[1]]$data()$x[1] |>
    materialize()

  task$data()$x[1] |>
    materialize()
  # Get original tensor for verification
  original = po("torch_ingress_ltnsr")$train(task)[[1]]
  original_tensor = as_torch_tensor(original)
  cat("Original tensor shape:", paste(dim(original_tensor), collapse="x"), "\n")
  
  # Get result tensor
  result = graph$predict(task)[[1]]
  result_tensor = as_torch_tensor(result)
  cat("Result tensor shape:", paste(dim(result_tensor), collapse="x"), "\n")
  
  # Explicitly check channel dimension
  expect_equal(dim(result_tensor)[2], dim(original_tensor)[2] - 1, 
               info = "Channel dimension should be reduced by 1")
})


test_that("nn_relu works on the lazy iris dataset", {
  po_sigmoid = po("nn_sigmoid")
  graph = po("torch_ingress_ltnsr") %>>% po_sigmoid

  task = tsk("lazy_iris")

  original_first_tnsr = task$data()$x[1] |> materialize()
  print(original_first_tnsr)

  trained = graph$train(task)
  trained_first_tnsr = trained[[1]]$task$data()$x[1] |> materialize()
  print(trained_first_tnsr)

  predicted = graph$train(task)
  predicted_first_tnsr = predicted[[1]]$task$data()$x[1] |> materialize()
  print(predicted_first_tnsr)
})

test_that("torch operation and PipeOp yield the same results", {
  # Create test tensor
  x = torch_randn(c(3, 4))
  print("Original tensor:")
  print(x)
  
  torch_result = torch_sigmoid(x)
  print("Direct torch result:")
  print(torch_result)
  
  task = as_task_regr(data.frame(y = 1:3), target = "y")
  task$cbind(lazy_tensor(x, "x"))
  
  # Create PipeOp graph with sigmoid
  graph = po("torch_ingress_ltnsr") %>>% po("nn_sigmoid")
  result = graph$train(task)[[1]]
  
  # Extract result tensor
  pipeop_result = as_torch_tensor(result)
  print("PipeOp result:")
  print(pipeop_result)
  
  # Verify results are identical
  expect_true(torch_allclose(torch_result, pipeop_result))
})
