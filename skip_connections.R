# task = tsk("mtcars")
#
# block1 = top("input") %>>% top("linear", out_features = 10, id = "linear1") %>>%
#   top("relu")
# block2 = top("input") %>>% top("linear", out_features = 10, id = "linear2")
#
# skip_connection = top("parallel", .paths = list(a = block1, b = block2), .reduce = "add")
#
# graph = top("input") %>>%
#   top("tokenizer", d_token = 10) %>>%
#   top("flatten", start_dim = 2, end_dim = 3) %>>%
#   skip_connection %>>%
#   top("linear", out_features = 1L) %>>%
#   top("model", n_epochs = 0L, optimizer = optim_adam, criterion = nn_mse_loss)
#
# learner = GraphLearner$new(graph, task_type = "regr")
# learner$train(task)
#
# # What is going on?
# # skip_connection(input) = path1(input) + path2(input)
# # the "+" is defined by the reduce operation
# # Moreover, the .paths and .reduce are NOT parameters but inherit the parameters of their paths
# skip_connection$param_set
