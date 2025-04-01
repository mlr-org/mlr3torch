devtools::load_all()

withr::local_options(mlr3torch.cache = TRUE)

# for the tiny imagenet data, should get only the blue channel
po = po("nn_fn", fn = function(x) x[, -1])
graph = po("torch_ingress_ltnsr") %>>% po

task = tsk("tiny_imagenet")$filter(1:10)
task_dt = task$data()

graph$train(task)
result = graph$predict(task)[[1]]
result_dt = result$data()

result_dt