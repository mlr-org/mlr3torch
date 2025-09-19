---
title: "Extracted Code from mlr3torch Paper"
author: "mlr3torch"
date: "2025-09-18"
output: html_document
---



``` r
options(mlr3torch.cache = TRUE)
lgr::get_logger("mlr3")$set_threshold("warn")
```


``` r
library("mlr3")
# make less verbose
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(42)
task <- tsk("mtcars")
learner <- lrn("regr.rpart")
split <- partition(task, ratio = 2/3)
learner$train(task, split$train)
pred <- learner$predict(task, split$test)
pred$score(msr("regr.rmse"))
```

```
## regr.rmse
##  4.736051
```

``` r
library("mlr3pipelines")
graph_learner <- as_learner(po("pca") %>>% lrn("regr.rpart"))

resampling <- rsmp("cv", folds = 3)
rr <- resample(task, graph_learner, resampling)
rr$aggregate(msr("regr.rmse"))
```

```
## regr.rmse
##  4.274766
```

``` r
library("torch")
torch_manual_seed(42)
x <- torch_tensor(1, device = "cpu")
w <- torch_tensor(2, requires_grad = TRUE, device = "cpu")
y <- w * x
y$backward()
w$grad
```

```
## torch_tensor
##  1
## [ CPUFloatType{1} ]
```

``` r
library("mlr3torch")
mnist <- tsk("mnist")
mnist
```

```
##
## ── <TaskClassif> (70000x2): MNIST Digit Classification ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
## • Target: label
## • Target classes: 1 (11%), 7 (10%), 3 (10%), 2 (10%), 9 (10%), 0 (10%), 6 (10%), 8 (10%), 4 (10%), 5 (9%)
## • Properties: multiclass
## • Features (1):
##   • lt (1): image
```

``` r
rows <- mnist$data(1:2)
rows
```

```
##     label           image
##    <fctr>   <lazy_tensor>
## 1:      5 <tnsr[1x28x28]>
## 2:      0 <tnsr[1x28x28]>
```

``` r
str(materialize(rows$image))
```

```
## List of 2
##  $ :Float [1:1, 1:28, 1:28]
##  $ :Float [1:1, 1:28, 1:28]
```

``` r
po_flat <- po("trafo_reshape", shape = c(-1, 28 * 28))
mnist_flat <- po_flat$train(list(mnist))[[1L]]
mnist_flat$head(2)
```

```
##     label         image
##    <fctr> <lazy_tensor>
## 1:      5   <tnsr[784]>
## 2:      0   <tnsr[784]>
```

``` r
mlp <- lrn("classif.mlp",
 loss = t_loss("cross_entropy"),
 optimizer = t_opt("adamw", lr = 0.001),
 callbacks = t_clbk("history"))

mlp$param_set$set_values(
  neurons = c(100, 200), activation = torch::nn_relu,
  p = 0.3, opt.weight_decay = 0.01, measures_train = msr("classif.logloss"),
  epochs = 10, batch_size = 32, device = "cuda")

mlp$configure(predict_type = "prob")

mlp$train(mnist_flat, row_ids = 1:60000)

mlp$model$network
```

```
## An `nn_module` containing 100,710 parameters.
##
## ── Modules ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
## • 0: <nn_linear> #78,500 parameters
## • 1: <nn_relu> #0 parameters
## • 2: <nn_dropout> #0 parameters
## • 3: <nn_linear> #20,200 parameters
## • 4: <nn_relu> #0 parameters
## • 5: <nn_dropout> #0 parameters
## • 6: <nn_linear> #2,010 parameters
```

``` r
head(mlp$model$callbacks$history, n = 2)
```

```
##    epoch train.classif.logloss
##    <num>                 <num>
## 1:     1             0.6931728
## 2:     2             0.4092444
```

``` r
pred <- mlp$predict(mnist_flat, row_ids = 60001:70000)
pred$score(msr("classif.ce"))
```

```
## classif.ce
##     0.0479
```

``` r
pth <- tempfile()
mlp$marshal()
saveRDS(mlp, pth)
mlp2 <- readRDS(pth)
mlp2$unmarshal()

set_validate(mlp, validate = 0.3)

nn_simple <- nn_module("nn_simple",
  initialize = function(d_in, d_latent, d_out) {
    self$linear1 = nn_linear(d_in, d_latent)
    self$activation = nn_relu()
    self$linear2 = nn_linear(d_latent, d_out)
  },
  forward = function(x) {
    x = self$linear1(x)
    x = self$activation(x)
    self$linear2(x)
  }
)

net <- nn_simple(10, 100, 1)

net(torch_randn(1, 10))
```

```
## torch_tensor
## 0.01 *
## -9.4603
## [ CPUFloatType{1,1} ][ grad_fn = <AddmmBackward0> ]
```

``` r
module_graph <- po("module_1", module = nn_linear(10, 100)) %>>%
 po("module_2", module = nn_relu()) %>>%
 po("module_3", module = nn_linear(100, 1))

net <- nn_graph(module_graph, shapes_in = list(module_1.input = c(NA, 10)))
net(torch_randn(2, 10))
```

```
## torch_tensor
## -0.1218
## -0.0636
## [ CPUFloatType{2,1} ][ grad_fn = <AddmmBackward0> ]
```

``` r
graph <- po("torch_ingress_ltnsr") %>>%
  nn("linear", out_features = 10) %>>% nn("relu") %>>% nn("head")

md <- graph$train(mnist_flat)[[1L]]
md
```

```
## <ModelDescriptor: 4 ops>
## * Ingress:  torch_ingress_ltnsr.input: [(NA,784)]
## * Task:  mnist [classif]
## * Callbacks:  N/A
## * Optimizer:  N/A
## * Loss:  N/A
## * pointer:  head.output [(NA,10)]
```

``` r
graph <- graph %>>%
  po("torch_loss", t_loss("cross_entropy")) %>>%
  po("torch_optimizer", t_opt("adamw", lr = 0.001))

graph <- graph %>>% po("torch_model_classif", epochs = 10, batch_size = 16)

glrn <- as_learner(graph)
glrn$train(mnist_flat)

path_lin <- nn("linear_1")
path_nonlin <- nn("linear_2") %>>% nn("relu")

residual_layer <- list(path_lin, path_nonlin) %>>% nn("merge_sum")

path_num <- po("select_1", selector = selector_type("numeric")) %>>%
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
    torch_clamp(nnf_mse_loss(input, target), max = self$max_loss)
  }
)
tloss <- as_torch_loss(nn_winsorized_mse)
tloss
```

```
## <TorchLoss:nn_winsorized_mse> nn_winsorized_mse
## * Generator: nn_winsorized_mse
## * Parameters: list()
## * Packages: torch,mlr3torch
## * Task Types: classif,regr
```

``` r
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
  forward = function(x) self$network(x)
)

num_input <- list(x = ingress_num())
num_input
```

```
## $x
## Ingress: Task[selector_type(c("numeric", "integer"))] --> Tensor()
```

``` r
lrn_ffn <- lrn("classif.module",
  module_generator = nn_ffn,
  ingress_tokens = num_input,
  latent_dim = 100, n_layers = 5)

task <- tsk("california_housing")
task
```

```
##
## ── <TaskRegr> (20640x10): California House Value ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
## • Target: median_house_value
## • Properties: -
## • Features (9):
##   • dbl (8): households, housing_median_age, latitude, longitude, median_income, population, total_bedrooms, total_rooms
##   • fct (1): ocean_proximity
```

``` r
preprocessing <- po("encode", method = "one-hot") %>>%
  po("imputehist")

ingress <- po("torch_ingress_num")

block <- nn("linear", out_features = 32) %>>%
  ppl("branch", list(relu = nn("relu"), sigmoid = nn("sigmoid"))) %>>%
  nn("dropout")

architecture <- nn("block", block) %>>% nn("head")

config <- po("torch_loss", loss = t_loss("mse")) %>>%
  po("torch_optimizer", optimizer = t_opt("adamw"))

model <- po("torch_model_regr", device = "cuda", batch_size = 512)

pipeline <- preprocessing %>>% ingress %>>%
  architecture %>>% config %>>% model
learner <- as_learner(pipeline)
learner$id <- "custom_nn"

library("mlr3tuning")
```

```
## Loading required package: paradox
```

``` r
learner$param_set$set_values(
  block.linear.out_features = to_tune(20, 500),
  block.n_blocks = to_tune(1, 5),
  block.branch.selection = to_tune(c("relu", "sigmoid")),
  block.dropout.p = to_tune(0.1, 0.9),
  torch_optimizer.lr = to_tune(10^-4, 10^-1, logscale = TRUE))

set_validate(learner, "test")

learner$param_set$set_values(
  torch_model_regr.patience = 5,
  torch_model_regr.measures_valid = msr("regr.mse"),
  torch_model_regr.epochs = to_tune(upper = 100, internal = TRUE))

library("mlr3mbo")
ti <- tune(
  tuner = tnr("mbo"),
  resampling = rsmp("holdout"),
  measure = msr("internal_valid_score", minimize = TRUE),
  learner = learner,
  term_evals = 40,
  task = task)
pvals <- ti$result_learner_param_vals[2:7]
cat(paste("*", names(pvals), "=", pvals,
 collapse = "\n"), "\n")
```

```
## * block.n_blocks = 4
## * block.linear.out_features = 394
## * block.branch.selection = relu
## * block.dropout.p = 0.526923747174442
## * torch_optimizer.lr = 0.000929559084734832
## * torch_model_regr.epochs = 98
```

``` r
library("torchdatasets")
dogs_vs_cats_dataset("data", download = TRUE)
```

```
## <dataset>
##   Public:
##     .getitem: function (i)
##     .length: function ()
##     classes: dog cat
##     clone: function (deep = FALSE)
##     images: data/dogs-vs-cats/train/cat.0.jpg data/dogs-vs-cats/trai ...
##     initialize: function (root, split = "train", download = FALSE, ..., transform = NULL,
##     load_state_dict: function (x, ..., .refer_to_state_dict = FALSE)
##     state_dict: function ()
##     target_transform: NULL
##     targets: 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2  ...
##     transform: NULL
```

``` r
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
```

```
## labels
##   cat   dog
## 12500 12500
```

``` r
tbl <- data.table(image = lt, class = labels)
task <- as_task_classif(tbl, target = "class", id = "dogs_vs_cats")
task
```

```
##
## ── <TaskClassif> (25000x2) ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
## • Target: class
## • Target classes: cat (positive class, 50%), dog (50%)
## • Properties: twoclass
## • Features (1):
##   • lt (1): image
```

``` r
augment <- po("augment_random_vertical_flip", p = 0.5)

preprocess <- po("trafo_resize", size = c(224, 224))

unfreezer <- t_clbk("unfreeze",
  starting_weights = select_name(c("fc.weight", "fc.bias")),
  unfreeze = data.table(epoch = 3, weights = select_all()))

resnet <- lrn("classif.resnet18",
  pretrained = TRUE, epochs = 5, device = "cuda", batch_size = 32,
  opt.lr = 1e-4, measures_valid = msr("classif.acc"),
  callbacks = list(unfreezer, t_clbk("history")))

learner <- as_learner(augment %>>% preprocess %>>% resnet)
learner$id <- "resnet"
set_validate(learner, 1 / 3)
learner$train(task)
learner$model$classif.resnet18$model$callbacks$history
```

```
##    epoch valid.classif.acc
##    <num>             <num>
## 1:     1         0.9504380
## 2:     2         0.9618385
## 3:     3         0.9765991
## 4:     4         0.9786391
## 5:     5         0.9804392
```

``` r
task <- tsk("melanoma")
task
```

```
##
## ── <TaskClassif> (32701x5): Melanoma Classification ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
## • Target: outcome
## • Target classes: malignant (positive class, 2%), benign (98%)
## • Properties: twoclass, groups
## • Features (4):
##   • fct (2): anatom_site_general_challenge, sex
##   • int (1): age_approx
##   • lt (1): image
## • Groups: patient_id
```

``` r
table(task$truth())
```

```
##
## malignant    benign
##       581     32120
```

``` r
task$missings("age_approx")
```

```
## age_approx
##         44
```

``` r
block_ffn <- nn("linear", out_features = 500) %>>%
  nn("relu") %>>% nn("dropout")
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
  nn("merge_cat") %>>% nn("linear_1", out_features = 500) %>>%
  nn("relu_4") %>>% nn("dropout_2") %>>% nn("head")

model <- architecture %>>%
  po("torch_loss",
    t_loss("cross_entropy", class_weight = torch_tensor(10))) %>>%
  po("torch_optimizer", t_opt("adamw", lr = 0.0005)) %>>%
  po("torch_model_classif", epochs = 4, batch_size = 32, device = "cuda",
    predict_type = "prob")

preprocessing <- po("classbalancing", ratio = 4, reference = "minor",
    adjust = "minor") %>>%
  po("augment_random_horizontal_flip") %>>%
  po("augment_random_vertical_flip") %>>%
  po("augment_random_crop", size = c(128, 128), pad_if_needed = TRUE)
glrn <- as_learner(preprocessing %>>% model)

library("mlr3viz")
glrn$id <- "multimodal"
rr <- resample(task, glrn, rsmp("cv", folds = 5))
plt <- autoplot(rr, type = "roc")
saveRDS(plt, here::here("paper/roc2.rds"))
```


``` r
sessionInfo()
```

```
## R version 4.5.0 (2025-04-11)
## Platform: x86_64-pc-linux-gnu
## Running under: Ubuntu 22.04.4 LTS
##
## Matrix products: default
## BLAS:   /usr/local/lib/R/lib/libRblas.so
## LAPACK: /usr/local/lib/R/lib/libRlapack.so;  LAPACK version 3.12.1
##
## locale:
##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C
##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8
##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8
##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C
##  [9] LC_ADDRESS=C               LC_TELEPHONE=C
## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C
##
## time zone: Etc/UTC
## tzcode source: system (glibc)
##
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base
##
## other attached packages:
##  [1] mlr3viz_0.10.1       torchdatasets_0.3.1  mlr3mbo_0.3.1
##  [4] mlr3tuning_1.4.0     paradox_1.0.1        mlr3torch_0.3.1.9000
##  [7] torch_0.16.0         future_1.67.0        mlr3pipelines_0.9.0
## [10] mlr3_1.2.0
##
## loaded via a namespace (and not attached):
##  [1] gtable_0.3.6         xfun_0.53            ggplot2_4.0.0
##  [4] processx_3.8.6       lattice_0.22-6       callr_3.7.6
##  [7] vctrs_0.6.5          tools_4.5.0          ps_1.9.1
## [10] safetensors_0.2.0    parallel_4.5.0       tibble_3.3.0
## [13] pkgconfig_2.0.3      Matrix_1.7-3         data.table_1.17.8
## [16] checkmate_2.3.3      RColorBrewer_1.1-3   S7_0.2.0
## [19] assertthat_0.2.1     uuid_1.2-1           lifecycle_1.0.4
## [22] farver_2.1.2         compiler_4.5.0       stringr_1.5.2
## [25] precrec_0.14.5       codetools_0.2-20     bbotk_1.6.0
## [28] pillar_1.11.1        crayon_1.5.3         rpart_4.1.24
## [31] parallelly_1.45.1    digest_0.6.37        stringi_1.8.7
## [34] listenv_0.9.1        mlr3measures_1.1.0   rprojroot_2.1.1
## [37] grid_4.5.0           here_1.0.2           cli_3.6.5
## [40] magrittr_2.0.4       future.apply_1.20.0  withr_3.0.2
## [43] scales_1.4.0         backports_1.5.0      rappdirs_0.3.3
## [46] bit64_4.6.0-1        spacefillr_0.4.0     globals_0.18.0
## [49] jpeg_0.1-11          bit_4.6.0            ranger_0.17.0
## [52] evaluate_1.0.5       knitr_1.50           torchvision_0.7.0
## [55] mlr3misc_0.19.0      rlang_1.1.6          Rcpp_1.1.0
## [58] zeallot_0.2.0        glue_1.8.0           palmerpenguins_0.1.1
## [61] coro_1.1.0           jsonlite_2.0.0       lgr_0.5.0
## [64] R6_2.6.1             fs_1.6.6             mlr3learners_0.12.0
```
