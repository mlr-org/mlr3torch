library("mlr3")
set.seed(42)
task <- tsk("mtcars")
learner <- lrn("regr.rpart")
split <- partition(task, ratio = 2/3)
learner$train(task, split$train)
pred <- learner$predict(task, split$test)
measure <- msr("regr.rmse")
pred$score(measure)

library("mlr3pipelines")
graph_learner <- as_learner(po("pca") %>>% lrn("regr.rpart"))

library("torch")
torch_manual_seed(42)
x <- torch_tensor(1, device = "cpu")
w <- torch_tensor(2, requires_grad = TRUE, device = "cpu")
y <- w * x
y$backward()
w$grad

library("mlr3torch")
mnist <- tsk("mnist")
mnist

rows <- mnist$data(1:2)
rows

str(materialize(rows$image))

po_flat <- po("trafo_reshape", shape = c(-1, 28 * 28))
mnist_flat <- po_flat$train(list(mnist))[[1L]]
mnist_flat$head(2)

library("mlr3torch")
mlp <- lrn("classif.mlp",
 loss = t_loss("cross_entropy"),
 optimizer = t_opt("adamw", lr = 0.01),
 callbacks = t_clbk("history")
)

mlp$param_set$set_values(
  epochs = 10, batch_size = 32, device = "cpu",
  neurons = c(100, 200), activation = torch::nn_relu,
  p = 0.3, opt.weight_decay = 0.01
)

mlp$configure(
  predict_type = "prob"
)

mlp$train(mnist_flat, row_ids = 1:60000)

mlp$model$network

pred <- mlp$predict(mnist_flat, row_ids = 60001:70000)
pred$score(msr("classif.ce"))

pth <- tempfile()
mlp$marshal()
saveRDS(mlp, pth)
mlp2 <- readRDS(pth)
mlp2$unmarshal()

set_validate(mlp, validate = 0.3)

module_graph <- po("module_1", module = nn_linear(10, 100)) %>>%
 po("module_2", module = nn_relu()) %>>%
 po("module_3", module = nn_linear(100, 1))

net <- nn_graph(module_graph, shapes_in = list(module_1.input = c(NA, 10)))
net(torch_randn(2, 10))

graph <- po("torch_ingress_ltnsr") %>>%
  nn("linear", out_features = 10) %>>%
  nn("relu") %>>%
  nn("head")

md <- graph$train(mnist_flat)[[1L]]
md

graph <- graph %>>%
  po("torch_loss", t_loss("cross_entropy")) %>>%
  po("torch_optimizer", t_opt("adamw", lr = 0.001))

graph <- graph %>>% po("torch_model_classif", epochs = 10, batch_size = 16)

glrn <- as_learner(graph)
glrn$train(mnist_flat)

path_lin <- nn("linear_1")
path_nonlin <- nn("linear_2") %>>% po("nn_relu")

residual_layer <- list(path_lin, path_nonlin) %>>% po("nn_merge_sum")
residual_layer

path_num <- po("select_1", selector = selector_type("numeric")) %>>%
  po("torch_ingress_num") %>>%
  po("nn_tokenizer_num")
path_categ <- po("select", selector = selector_type("factor")) %>>%
  po("torch_ingress_categ") %>>%
  po("nn_tokenizer_categ")

graph <- list(path_num, path_categ) %>>% po("nn_merge_cat")

blocks <- nn("block", residual_layer, n_blocks = 5)

tloss <- as_torch_loss(torch::nn_l1_loss)
tloss

topt <- as_torch_optimizer(torch::optim_adam)
topt


gradient_clipper <- torch_callback("gradient_clipper",
  initialize = function(max_norm, norm_type) {
    self$norms <- numeric()
    self$max_norm <- max_norm
    self$norm_type <- norm_type
  },
  on_after_backward = function() {
    norm <- nn_utils_clip_grad_norm_(self$ctx$network$parameters,
      self$max_norm, self$norm_type)
    self$norms <- c(self$norms, norm$item())
  },
  state_dict = function() {
    self$norms
  },
  load_state_dict = function(state_dict) {
    self$norms = state_dict
  }
)
gradient_clipper

nn_ffn <- nn_module("nn_ffn",
  initialize = function(task, latent_dim, n_layers) {
    dims <- c(task$n_features, rep(latent_dim, n_layers),
      length(task$class_names))
    modules <- unlist(lapply(seq_len(length(dims) - 1), function(i) {
      if (i < length(dims) - 1) {
        list(nn_linear(dims[i], dims[i + 1]), nn_relu())
      } else {
        list(nn_linear(dims[i], dims[i + 1]))
      }
    }), recursive = FALSE)
    self$network <- do.call(nn_sequential, modules)
  },
  forward = function(x) {
    self$network(x)
  }
)

num_input <- list(x = ingress_num())
num_input

lrn_ffn <- lrn("classif.module",
  module_generator = nn_ffn,
  ingress_tokens = num_input,
  latent_dim = 100, n_layers = 5
)

library("mlr3data")
task <- tsk("california_housing")
task

preprocessing <- po("encode", method = "one-hot") %>>%
  po("imputehist")

ingress <- po("torch_ingress_num")

block <- nn("linear", out_features = 32) %>>%
  ppl("branch", list(relu = nn("relu"), tanh = nn("sigmoid"))) %>>%
  nn("dropout")

architecture <- nn("block", block) %>>%
  nn("head")

config <- po("torch_loss", loss = t_loss("mse")) %>>%
  po("torch_optimizer", optimizer = t_opt("adamw"))

model <- po("torch_model_regr", device = "cuda", batch_size = 128)

pipeline <- preprocessing %>>%
  ingress %>>%
  architecture %>>%
  config %>>%
  model
learner = as_learner(pipeline)
learner$id = "custom_nn"

library("mlr3tuning")
learner$param_set$set_values(
  block.linear.out_features = to_tune(20, 500),
  block.n_blocks = to_tune(1, 10),
  block.branch.selection = to_tune(c("relu", "tanh")),
  block.dropout.p = to_tune(0.1, 0.9),
  torch_optimizer.lr = to_tune(10^-4, 10^-1, logscale = TRUE)
)

set_validate(learner, "test")

learner$param_set$set_values(
  torch_model_regr.patience = 10,
  torch_model_regr.measures_valid = msr("regr.mse"),
  torch_model_regr.epochs = to_tune(upper = 100, internal = TRUE)
)

library("mlr3mbo")
ti <- tune(
  tuner = tnr("mbo"),
  resampling = rsmp("cv", folds = 3),
  measure = msr("internal_valid_score", minimize = TRUE),
  learner = learner,
  term_evals = 100,
  task = task
)
ti

library("torchdatasets")
data_dir = here::here("data")
dogs_vs_cats_dataset(data_dir, download = TRUE)

ds <- torch::dataset("dogs_vs_cats",
  initialize = function(pths) {
    self$pths <- pths
  },
  .getitem = function(i) {
    image <- torchvision::base_loader(self$pths[i])
    list(image = torch_tensor(image)$permute(c(3, 1, 2)))
  },
  .length = function() {
    length(self$pths)
  }
)

paths <- list.files(file.path(data_dir, "dogs-vs-cats/train"),
  full.names = TRUE)
dogs_vs_cats <- ds(paths)

lt <- as_lazy_tensor(dogs_vs_cats, list(image = NULL))

labels <- ifelse(grepl("dog\\.\\d+\\.jpg", paths), "dog", "cat")
table(labels)

tbl <- data.table(image = lt, class = labels)
task <- as_task_classif(tbl, target = "class", id = "dogs_vs_cats")
task

augment <- po("augment_random_vertical_flip", p = 0.5)

preprocess <- po("trafo_resize", size = c(224, 224))

unfreezer <- t_clbk("unfreeze",
  starting_weights = select_name(c("fc.weight", "fc.bias")),
  unfreeze = data.table(
    epoch = 3, weights = select_all()
  )
)

resnet <- lrn("classif.resnet18",
  pretrained = TRUE, epochs = 10,
  device = "cuda", batch_size = 32,
  measures_valid = msr("classif.acc"),
  callbacks = list(unfreezer, t_clbk("history"))
)

library("ggplot2")
learner <- as_learner(augment %>>% preprocess %>>% resnet)
learner$id <- "resnet"
set_validate(learner, 1 / 3)
learner$train(task)
history <- learner$model$classif.alexnet$callbacks$history
ggplot(history, aes(x = epoch, y = valid.classif.acc)) +
  geom_point()

task <- tsk("melanoma")
task

table(task$truth())

task$missings("age_approx")

block_ffn <- nn("linear", out_features = 500) %>>%
  nn("relu") %>>%
  nn("dropout")
path_tabular <- po("select_1",
    selector = selector_type(c("integer", "factor"))) %>>%
  po("imputehist") %>>%
  po("encode", method = "one-hot") %>>%
  po("torch_ingress_num") %>>%
  nn("block_1", block = block_ffn, n_blocks = 3)

path_image <- po("select_2", selector = selector_name("image")) %>>%
  po("torch_ingress_ltnsr", shape = c(NA, 3, 128, 128)) %>>%
  nn("conv2d_1", out_channels = 64, kernel_size = 7, stride = 2,
    padding = 3) %>>%
  nn("batch_norm2d_1") %>>%
  nn("relu_1") %>>%
  nn("max_pool2d_1", kernel_size = 3, stride = 2, padding = 1) %>>%
  nn("conv2d_2", out_channels = 128, kernel_size = 3, stride = 1,
    padding = 1) %>>%
  nn("batch_norm2d_2") %>>%
  nn("relu_2") %>>%
  nn("conv2d_3", out_channels = 256, kernel_size = 3, stride = 1,
    padding = 1) %>>%
  nn("batch_norm2d_3") %>>%
  nn("relu_3") %>>%
  nn("flatten")

architecture <- list(path_tabular, path_image) %>>%
  nn("merge_cat") %>>%
  nn("linear_1", out_features = 500) %>>%
  nn("relu_4") %>>%
  nn("dropout_2") %>>%
  nn("head")

model <- architecture %>>%
  po("torch_loss",
    t_loss("cross_entropy", weight = torch_tensor(c(10, 1)))) %>>%
  po("torch_optimizer", t_opt("adamw", lr = 0.0005)) %>>%
  po("torch_model_classif", epochs = 4, batch_size = 32, device = "cuda",
    predict_type = "prob")

preprocessing <- po("classbalancing", ratio = 4, reference = "minor",
    adjust = "minor") %>>%
  po("augment_random_horizontal_flip") %>>%
  po("augment_random_vertical_flip") %>>%
  po("augment_random_crop", size = c(128, 128), pad_if_needed = TRUE)
glrn <- as_learner(preprocessing %>>% model)

glrn$id <- "multimodal"
rr <- resample(task, glrn, rsmp("cv", folds = 5))
autoplot(rr, type = "roc")

