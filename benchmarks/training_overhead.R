# Overhead of mlr3torch per batch compared to plain R torch, on CPU with one thread.
# Results at the bottom: "before" is mlr3torch 8b69e64e, "after" the commit that added this file
# (forward plan for nn_graph and lazy tensor preprocessing, batch-wise preprocessing, cache_task_data).
# Run with OMP_NUM_THREADS=1 in the CPU docker image of the mlr3torch paper (torch 0.17.0).
devtools::load_all(".")
lgr::get_logger("mlr3")$set_threshold("warn")
torch_set_num_threads(1)
set.seed(1)
torch_manual_seed(1)

# median time in seconds of `reps` runs of `f()`
reps = 5L
secs = function(f) {
  median(replicate(reps, {
    t0 = Sys.time()
    f()
    as.numeric(difftime(Sys.time(), t0, units = "secs"))
  }))
}
report = function(what, s, n_batches) {
  cat(sprintf("%-60s %8.2f s %9.2f ms/batch\n", what, s, 1000 * s / n_batches))
}

# Tabular data: MLP on 20000 x 20, batch size 32, one epoch
n = 20000L
p = 20L
bs = 32L
n_batches = ceiling(n / bs)
X = matrix(rnorm(n * p), n, p, dimnames = list(NULL, paste0("x", seq_len(p))))
y = drop(X %*% rnorm(p)) + rnorm(n)
task = as_task_regr(data.table(X, y = y), target = "y", id = "synthetic")

make_net = function() nn_sequential(nn_linear(p, 64), nn_relu(), nn_linear(64, 64), nn_relu(), nn_linear(64, 1))
Xt = torch_tensor(X)
yt = torch_tensor(matrix(y))
plain_torch = function() {
  net = make_net()
  opt = optim_ignite_adamw(net$parameters, lr = 1e-3)
  perm = sample(n)
  for (i in seq_len(n_batches)) {
    idx = perm[((i - 1) * bs + 1):min(i * bs, n)]
    opt$zero_grad()
    loss = nnf_mse_loss(net(Xt[idx, ]), yt[idx, ])
    loss$backward()
    opt$step()
  }
}
mlp = function(...) {
  lrn("regr.mlp", neurons = c(64, 64), batch_size = bs, epochs = 1, device = "cpu",
    optimizer = t_opt("adamw", lr = 1e-3), ...)
}
report("plain R torch", secs(plain_torch), n_batches)
report("regr.mlp", secs(function() mlp()$train(task)), n_batches)
report("regr.mlp, tensor_dataset = TRUE", secs(function() mlp(tensor_dataset = TRUE)$train(task)), n_batches)
report("regr.mlp, measures_train + history",
  secs(function() mlp(measures_train = msr("regr.mse"), callbacks = t_clbk("history"))$train(task)), n_batches)
learner = mlp()
learner$train(task)
report("regr.mlp, predict", secs(function() learner$predict(task)), n_batches)

# Lazy tensors with preprocessing: MLP on 20000 images of size 1 x 28 x 28, batch size 32, one epoch
images = torch_randint(0, 256, c(n, 1, 28, 28))$to(dtype = torch_float())
task_images = as_task_classif(data.table(image = as_lazy_tensor(images), label = factor(sample(0:9, n, TRUE))),
  target = "label", id = "images")
task_flat = (po("trafo_normalize", mean = 0, std = 255) %>>%
  po("trafo_reshape", shape = c(-1, 784)))$train(task_images)[[1L]]
classif_mlp = function(...) {
  lrn("classif.mlp", neurons = c(100, 200), batch_size = bs, epochs = 1, device = "cpu", ...)
}
report("classif.mlp, lazy tensor", secs(function() classif_mlp()$train(task_flat)), n_batches)
report("classif.mlp, lazy tensor, tensor_dataset = TRUE",
  secs(function() classif_mlp(tensor_dataset = TRUE)$train(task_flat)), n_batches)
cat(sprintf("%-60s %8.2f s\n", "materialize() of 20000 preprocessed images",
  secs(function() materialize(task_flat$data(cols = "image")$image, rbind = TRUE))))

# A network built from PipeOpTorch operators: attention network on California housing, batch size 256
data = as.data.table(tsk("california_housing"))
data[, median_house_value := median_house_value / 1e5]
task_cal = as_task_regr(data, target = "median_house_value", id = "california")
block = nn("multihead_attention", num_heads = 1) %>>% nn("layer_norm", dims = 1) %>>%
  nn("linear", out_features = 32) %>>% nn("relu") %>>% nn("dropout", p = 0.3)
make_learner_cal = function() as_learner_torch(po("imputemedian") %>>% po("encode", method = "one-hot") %>>% po("scale") %>>%
  po("torch_ingress_num") %>>% nn("tokenizer_num", d_token = 8) %>>% nn("block", block, n_blocks = 1) %>>%
  nn("flatten") %>>% nn("head") %>>% po("torch_loss", t_loss("mse")) %>>% po("torch_optimizer", t_opt("adamw")) %>>%
  po("torch_model_regr", epochs = 1, batch_size = 256, device = "cpu"))
report("attention network (GraphLearner), batch size 256", secs(function() make_learner_cal()$train(task_cal)),
  ceiling(task_cal$nrow / 256))

# Results: median of 5 runs, ms per batch (seconds for materialize()), one pinned CPU core each.
# before: mlr3torch 8b69e64e, the commit before the performance changes
#                                                     before    after
# plain R torch                                         3.38     3.31
# regr.mlp                                              8.86     7.17
# regr.mlp, tensor_dataset = TRUE                       5.58     5.69
# regr.mlp, measures_train + history                   11.15     9.25
# regr.mlp, predict                                     6.24     4.13
# classif.mlp, lazy tensor                            120.41    14.01
# classif.mlp, lazy tensor, tensor_dataset = TRUE      74.36     9.66
# materialize() of 20000 preprocessed images (s)       41.86     0.17
# attention network (GraphLearner), batch size 256    179.59    49.83
