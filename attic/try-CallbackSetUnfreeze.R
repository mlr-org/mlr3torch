devtools::load_all()

task = tsk("iris")

mlp = lrn("classif.mlp",
          epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
)

# sela = selector_all()
# sela(mlp$network$modules)

mlp$train(task)

# do this for each element in the parameters list
mlp$model$network$modules[["9"]]$parameters[[1]]$requires_grad_(TRUE)
mlp$model$network$modules[["9"]]$parameters[[2]]$requires_grad_(TRUE)


# construct a NN as a graph
module_1 = nn_linear(in_features = 3, out_features = 4, bias = TRUE)
activation = nn_sigmoid()
module_2 = nn_linear(4, 3, bias = TRUE)
softmax = nn_softmax(2)

po_module_1 = po("module_1", module = module_1)
po_activation = po("module", id = "activation", activation)
po_module_2 = po("module_2", module = module_2)
po_softmax = po("module", id = "softmax", module = softmax)

module_graph = po_module_1 %>>%
  po_activation %>>%
  po_module_2 %>>%
  po_softmax

module_graph$plot(html = TRUE)

module_graph