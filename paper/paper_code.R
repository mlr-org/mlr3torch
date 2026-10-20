#' ```{r setup, include=FALSE}
#' knitr::opts_chunk$set(cache = FALSE)
#' ```
Sys.time()
options(width = 70, useFancyQuotes = FALSE)
options(mlr3torch.cache = TRUE)
lgr::get_logger("mlr3")$set_threshold("warn")
library("mlr3torch")
set.seed(42)
torch_manual_seed(42)
mnist <- tsk("mnist")
mnist

rows <- mnist$data(1:2)
rows

str(materialize(rows$image))

po_flat <- po("trafo_reshape", shape = c(-1, 28 * 28))
mnist_flat <- po_flat$train(list(mnist))[[1L]]
mnist_flat$head(2)

str(materialize(mnist_flat$head(2)$image))

mlp <- lrn("classif.mlp",
  loss = t_loss("cross_entropy"),
  optimizer = t_opt("adamw", lr = 0.001),
  callbacks = t_clbk("history"))

mlp$param_set$set_values(
  neurons = c(100, 200), activation = nn_relu,
  p = 0.3, opt.weight_decay = 0.01,
  measures_train = msr("classif.logloss"),
  epochs = 10, batch_size = 32, device = "cpu")

mlp$configure(predict_type = "prob")

mlp$train(mnist_flat, row_ids = 1:1000)

mlp$model$network

head(mlp$model$callbacks$history, n = 2)

pred <- mlp$predict(mnist_flat, row_ids = 1001:1100)
pred$score(msr("classif.ce"))

mnist_small <- mnist_flat$clone()$filter(1:1000)
rr <- resample(mnist_small, mlp, rsmp("cv", folds = 3))
rr$aggregate(msr("classif.ce"))

pth <- tempfile()
mlp$marshal()
saveRDS(mlp, pth)
mlp2 <- readRDS(pth)
mlp2$unmarshal()

set_validate(mlp, validate = 0.3)

nn_simple <- nn_module("nn_simple",
  initialize = function(d_in, d_latent, d_out) {
    self$linear1 <- nn_linear(d_in, d_latent)
    self$activation <- nn_relu()
    self$linear2 <- nn_linear(d_latent, d_out)
  },
  forward = function(x) {
    x <- self$linear1(x)
    x <- self$activation(x)
    self$linear2(x)
  }
)

net <- nn_simple(10, 100, 1)

net(torch_randn(1, 10))

module_graph <- po("module_1", module = nn_linear(10, 100)) %>>%
  po("module_2", module = nn_relu()) %>>%
  po("module_3", module = nn_linear(100, 1))

net <- nn_graph(module_graph,
  shapes_in = list(module_1.input = c(NA, 10)))
net(torch_randn(2, 10))

graph <- po("torch_ingress_ltnsr") %>>%
  nn("linear", out_features = 100) %>>% nn("relu") %>>% nn("head")

md <- graph$train(mnist_flat)[[1L]]
md

graph <- graph %>>%
  po("torch_loss", t_loss("cross_entropy")) %>>%
  po("torch_optimizer", t_opt("adamw", lr = 0.001))

graph <- graph %>>%
  po("torch_model_classif", epochs = 10, batch_size = 16, device = "cpu")

glrn <- as_learner_torch(graph)
glrn$train(mnist_flat, row_ids = 1:1000)

path_lin <- nn("linear_1")
path_nonlin <- nn("linear_2") %>>% nn("relu")

residual_layer <- list(path_lin, path_nonlin) %>>% nn("merge_sum")

path_num <- po("select_1",
  selector = selector_type(c("numeric", "integer"))) %>>%
  po("torch_ingress_num") %>>%
  nn("tokenizer_num", d_token = 10)
path_categ <- po("select_2", selector = selector_type("factor")) %>>%
  po("torch_ingress_categ") %>>%
  nn("tokenizer_categ", d_token = 10)

graph <- list(path_num, path_categ) %>>% nn("merge_cat", dim = 2)

blocks <- nn("block", residual_layer, n_blocks = 5)

nn_winsorized_mse <- nn_module(c("nn_winsorized_mse", "nn_loss"),
  initialize = function(max_loss) {
    self$max_loss <- max_loss
  },
  forward = function(input, target) {
    torch_mean(torch_clamp((input - target)^2, max = self$max_loss))
  }
)
tloss <- as_torch_loss(nn_winsorized_mse)
tloss

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
    self$norms <- state_dict
  }
)

nn_ffn <- nn_module("nn_ffn",
  initialize = function(task, latent_dim, n_layers) {
    dims <- c(task$n_features, rep(latent_dim, n_layers),
      output_dim_for(task))
    modules <- unlist(lapply(seq_len(length(dims) - 1), function(i) {
      if (i < length(dims) - 1) {
        list(nn_linear(dims[i], dims[i + 1]), nn_relu())
      } else {
        list(nn_linear(dims[i], dims[i + 1]))
      }
    }), recursive = FALSE)
    self$network <- do.call(nn_sequential, modules)
  },
  forward = function(x) self$network(x)
)

num_input <- list(x = ingress_num())
num_input

lrn_ffn <- lrn("classif.module",
  module_generator = nn_ffn,
  ingress_tokens = num_input,
  latent_dim = 100, n_layers = 5)

task <- tsk("mtcars")
task

ingress <- po("torch_ingress_num")

embedding <- nn("tokenizer_num", d_token = 8)

block <- nn("multihead_attention", num_heads = 2) %>>%
  nn("layer_norm", dims = 1) %>>%
  nn("linear", out_features = 16) %>>%
  ppl("branch", list(relu = nn("relu"), sigmoid = nn("sigmoid"))) %>>%
  nn("dropout")

architecture <- embedding %>>% nn("block", block) %>>%
  nn("flatten") %>>% nn("head")

config <- po("torch_loss", loss = t_loss("mse")) %>>%
  po("torch_optimizer", optimizer = t_opt("adamw"))

model <- po("torch_model_regr", device = "cpu", batch_size = 32)

pipeline <- ingress %>>%
  architecture %>>% config %>>% model
learner <- as_learner_torch(pipeline)
learner$id <- "attention_nn"

library("mlr3tuning")
learner$param_set$set_values(
  tokenizer_num.d_token = to_tune(c(4, 8, 16)),
  block.multihead_attention.num_heads = to_tune(c(1, 2, 4)),
  block.multihead_attention.dropout = to_tune(0, 0.5),
  block.linear.out_features = to_tune(c(8, 16, 32)),
  block.n_blocks = to_tune(1, 3),
  block.branch.selection = to_tune(c("relu", "sigmoid")),
  block.dropout.p = to_tune(0.1, 0.9),
  torch_optimizer.lr = to_tune(10^-4, 10^-1, logscale = TRUE))

set_validate(learner, "test")

learner$param_set$set_values(
  torch_model_regr.patience = 5,
  torch_model_regr.measures_valid = msr("regr.mse"),
  torch_model_regr.epochs = to_tune(upper = 100, internal = TRUE))

ti <- tune(
  tuner = tnr("random_search"),
  resampling = rsmp("holdout"),
  measure = msr("internal_valid_score", minimize = TRUE),
  learner = learner,
  term_evals = 20,
  task = task)
tuned <- learner$param_set$search_space()$ids()
pvals <- lapply(ti$result_learner_param_vals[tuned],
  function(x) if (is.numeric(x)) signif(x, 3) else x)
cat(paste("*", names(pvals), "=", pvals,
  collapse = "\n"), "\n")

library("torchdatasets")
dvc <- dogs_vs_cats_dataset("data", download = TRUE)

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

paths <- list.files(file.path("data", "dogs-vs-cats/train"),
  full.names = TRUE)
dogs_vs_cats <- ds(paths)

lt <- as_lazy_tensor(dogs_vs_cats, list(image = NULL))

labels <- ifelse(grepl("dog\\.\\d+\\.jpg", paths), "dog", "cat")
table(labels)

tbl <- data.table(image = lt, class = labels)
task <- as_task_classif(tbl, target = "class", id = "dogs_vs_cats")
task

augment <- po("augment_random_horizontal_flip", p = 0.5)

preprocess <- po("trafo_resize", size = c(224, 224))

normalize <- po("trafo_normalize",
  mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))

unfreezer <- t_clbk("unfreeze",
  starting_weights = select_name(c("head.weight", "head.bias")),
  unfreeze = data.table(epoch = 3, weights = select_all()))

convnext <- lrn("classif.convnext_tiny_1k",
  pretrained = TRUE, epochs = 5, device = "cpu", batch_size = 32,
  opt.lr = 1e-4, measures_valid = msr("classif.acc"),
  callbacks = list(unfreezer, t_clbk("history")))

learner <- as_learner(
  augment %>>% preprocess %>>% normalize %>>% convnext)
learner$id <- "convnext"
set_validate(learner, 1 / 3)
learner$train(task, 12401:12600)
learner$model$classif.convnext_tiny_1k$model$callbacks$history

task <- tsk("melanoma")
task

table(task$truth())

task$missings("age_approx")

path_tabular <- po("select_1",
    selector = selector_type(c("integer", "factor"))) %>>%
  po("imputehist") %>>%
  po("encode", method = "one-hot") %>>%
  po("torch_ingress_num") %>>%
  nn("tokenizer_num", d_token = 64)

patchify <- function(x) {
  nnf_unfold(x, kernel_size = 16, stride = 16)$transpose(2, 3)
}
path_image <- po("select_2", selector = selector_name("image")) %>>%
  po("torch_ingress_ltnsr", shape = c(NA, 3, 128, 128)) %>>%
  nn("fn_1", fn = patchify) %>>%
  nn("linear", out_features = 64)

po_torch_pos_embedding <- pipeop_torch("nn_pos_embedding",
  initialize = function(shapes_in) {
    self$weight <- nn_parameter(torch_randn(shapes_in[[1]][-1]) * 0.02)
  },
  forward = function(input) input + self$weight,
  shapes_out = function(shapes_in) shapes_in
)
path_image <- path_image %>>% po_torch_pos_embedding$new()

architecture <- list(path_tabular, path_image) %>>%
  nn("merge_cat", dim = 2) %>>%
  nn("block", nn("transformer_encoder_layer", nhead = 4,
    dim_feedforward = 128), n_blocks = 2) %>>%
  nn("fn_2", fn = function(x) x$mean(dim = 2)) %>>%
  nn("head")

model <- architecture %>>%
  po("torch_loss",
    t_loss("cross_entropy", class_weight = torch_tensor(10))) %>>%
  po("torch_optimizer", t_opt("adamw", lr = 0.0005)) %>>%
  po("torch_model_classif", epochs = 4, batch_size = 32, device = "cpu",
    predict_type = "prob")

preprocessing <- po("classbalancing", ratio = 4, reference = "minor",
    adjust = "minor") %>>%
  po("augment_random_horizontal_flip") %>>%
  po("augment_random_vertical_flip") %>>%
  po("augment_random_crop", size = c(128, 128), padding = 8)
glrn <- as_learner_torch(preprocessing %>>% model)

library("mlr3learners")
baseline <- as_learner(
  po("select",
    selector = selector_invert(selector_type("lazy_tensor"))) %>>%
  po("imputehist") %>>% po("encode", method = "treatment") %>>%
  lrn("classif.log_reg", predict_type = "prob"))
baseline$id <- "tabular"
glrn$id <- "multimodal"
task_subset <- task$clone(deep = TRUE)
ids <- partition(task_subset, ratio = 0.1)$train
task_subset$filter(ids)
design <- benchmark_grid(task_subset, list(glrn, baseline),
  rsmp("holdout"))
bmr <- benchmark(design)
library("mlr3viz")
plt <- autoplot(bmr, type = "roc")

library("ggplot2")
plt = plt +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )
print(plt)
ggsave(here::here("roc.png"), plt, width = 4, height = 4, dpi = 300)
Sys.time()
sessionInfo()
