test_that("multiplication works", {
  task = tsk("pima")
  po_input = PipeOpInput$new()
  out = po_input$train(list(task))
  po_linear = PipeOpLinear$new(param_vals = list(units = 10))
  output = po_linear$train(out)

  po_model = PipeOpModel$new()
  x = po_input$train(list(task))
  po_linear$train(x)

  graph = po_input %>>% po_linear



  graph$train(list(task))

})
