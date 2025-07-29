---
title: "Extracted Code from mlr3torch Paper"
author: "mlr3torch"
date: "2025-07-25"
output: html_document
---


``` r
library("mlr3")
set.seed(42)
task <- tsk("mtcars")
learner <- lrn("regr.rpart")
split <- partition(task, ratio = 2/3)
learner$train(task, split$train)
pred <- learner$predict(task, split$test)
rmse <- msr("regr.rmse")
pred$score(rmse)
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
rr$aggregate(rmse)
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
## ── <TaskClassif> (70000x2): MNIST Digit Classification ─────────────────────────
## • Target: label
```

```
## Dataset <mnist> (~12 MB) will be downloaded and processed if not already
## available.
## Downloading <mnist> ...
## Processing <mnist>...
## Dataset <mnist> downloaded and extracted successfully.
## Dataset <mnist> loaded with 60000 images.
## Dataset <mnist> loaded with 10000 images.
```

```
## • Target classes: 1 (11%), 7 (10%), 3 (10%), 2 (10%), 9 (10%), 0 (10%), 6
## (10%), 8 (10%), 4 (10%), 5 (9%)
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
 optimizer = t_opt("adamw", lr = 0.01),
 callbacks = t_clbk("history")
)

mlp$param_set$set_values(
  neurons = c(100, 200), activation = torch::nn_relu,
  p = 0.3, opt.weight_decay = 0.01, measures_train = msr("classif.logloss"),
  epochs = 5, batch_size = 32, device = "cpu"
)

mlp$configure(
  predict_type = "prob",
  epochs = 10
)

mlp$train(mnist_flat, row_ids = 1:60000)

mlp$model$network
```

```
## An `nn_module` containing 100,710 parameters.
## 
## ── Modules ─────────────────────────────────────────────────────────────────────
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
## 1:     1              2.299611
## 2:     2              2.307922
```

``` r
pred <- mlp$predict(mnist_flat, row_ids = 60001:70000)
pred$score(msr("classif.ce"))
```

```
## classif.ce 
##      0.902
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
  nn("linear", out_features = 10) %>>%
  nn("relu") %>>%
  nn("head")

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
    loss <- nnf_mse_loss(input, target)
    loss <- torch_clamp(loss, max = self$max_loss)
    loss
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
  forward = function(x) {
    self$network(x)
  }
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
  latent_dim = 100, n_layers = 5
)

task <- tsk("california_housing")
task
```

```
## 
## ── <TaskRegr> (20640x10): California House Value ───────────────────────────────
## • Target: median_house_value
## • Properties: -
## • Features (9):
##   • dbl (8): households, housing_median_age, latitude, longitude,
##   median_income, population, total_bedrooms, total_rooms
##   • fct (1): ocean_proximity
```

``` r
preprocessing <- po("encode", method = "one-hot") %>>%
  po("imputehist")

ingress <- po("torch_ingress_num")

block <- nn("linear", out_features = 32) %>>%
  ppl("branch", list(relu = nn("relu"), sigmoid = nn("sigmoid"))) %>>%
  nn("dropout")

architecture <- nn("block", block) %>>%
  nn("head")

config <- po("torch_loss", loss = t_loss("mse")) %>>%
  po("torch_optimizer", optimizer = t_opt("adamw"))

model <- po("torch_model_regr", device = "cuda", batch_size = 512)

pipeline <- preprocessing %>>%
  ingress %>>%
  architecture %>>%
  config %>>%
  model
learner = as_learner(pipeline)
learner$id = "custom_nn"

library("mlr3tuning")
```

```
## Loading required package: paradox
```

``` r
learner$param_set$set_values(
  block.linear.out_features = to_tune(20, 500),
  block.n_blocks = to_tune(1, 5),
  block.branch.selection = to_tune(c("relu", "tanh")),
  block.dropout.p = to_tune(0.1, 0.9),
  torch_optimizer.lr = to_tune(10^-4, 10^-1, logscale = TRUE)
)
```

```
## Error in self$assert(xs, sanitize = TRUE): Assertion on 'xs' failed: tune token invalid: to_tune(c("relu", "tanh")) generates points that are not compatible with param block.branch.selection.
## Bad value:
## [1] "tanh"
## Parameter:
## Param of class "ParamFct":
## 
##                        id      cls         grouping  cargo lower upper
##                    <char>   <char>           <char> <list> <num> <num>
## 1: block.branch.selection ParamFct "relu","sigmoid" [NULL]    NA    NA
##    tolerance       levels special_vals        default storage_type
##        <num>       <list>       <list>         <list>       <char>
## 1:        NA relu,sigmoid    <list[0]> <NoDefault[0]>    character.
```

``` r
set_validate(learner, "test")

learner$param_set$set_values(
  torch_model_regr.patience = 5,
  torch_model_regr.measures_valid = msr("regr.mse"),
  torch_model_regr.epochs = to_tune(upper = 100, internal = TRUE)
)

library("mlr3mbo")
ti <- tune(
  tuner = tnr("mbo"),
  resampling = rsmp("holdout"),
  measure = msr("internal_valid_score", minimize = TRUE),
  learner = learner,
  term_evals = 40,
  task = task
)
```

```
## Error in .__ParamSet__get_values(self = self, private = private, super = super, : Missing required parameters: block.n_blocks
```

``` r
ti$result_learner_param_vals[2:7]
```

```
## Error in ti$result_learner_param_vals: object of type 'closure' is not subsettable
```

``` r
library("torchdatasets")
data_dir = "data"
dogs_vs_cats_dataset(data_dir, download = TRUE)
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

paths <- list.files(file.path(data_dir, "dogs-vs-cats/train"),
  full.names = TRUE)
dogs_vs_cats <- ds(paths)

lt <- as_lazy_tensor(dogs_vs_cats, list(image = NULL))

labels <- ifelse(grepl("dog\\.\\d+\\.jpg", paths), "dog", "cat")
table(labels)
```

```
## labels
##   cat   dog 
## 12500  5490
```

``` r
tbl <- data.table(image = lt, class = labels)
task <- as_task_classif(tbl, target = "class", id = "dogs_vs_cats")
task
```

```
## 
## ── <TaskClassif> (17990x2) ─────────────────────────────────────────────────────
## • Target: class
## • Target classes: cat (positive class, 69%), dog (31%)
## • Properties: twoclass
## • Features (1):
##   • lt (1): image
```

``` r
augment <- po("augment_random_vertical_flip", p = 0.5)

preprocess <- po("trafo_resize", size = c(224, 224))

unfreezer <- t_clbk("unfreeze",
  starting_weights = select_name(c("fc.weight", "fc.bias")),
  unfreeze = data.table(
    epoch = 3, weights = select_all()
  )
)

resnet <- lrn("classif.resnet18",
  pretrained = TRUE, epochs = 5,
  device = "cuda", batch_size = 32,
  opt.lr = 1e-4,
  measures_valid = msr("classif.acc"),
  callbacks = list(unfreezer, t_clbk("history"))
)

library("ggplot2")
learner <- as_learner(augment %>>% preprocess %>>% resnet)
learner$id <- "resnet"
set_validate(learner, 1 / 3)
learner$train(task)
```

```
## Error in jpeg::readJPEG(path): JPEG decompression error: Empty input file
## This happened PipeOp classif.resnet18's $train()
```

``` r
history <- learner$model$classif.resnet18$model$callbacks$history
ggplot(history, aes(x = epoch, y = valid.classif.acc)) +
  geom_point()
```

```
## Error in `geom_point()`:
## ! Problem while computing aesthetics.
## ℹ Error occurred in the 1st layer.
## Caused by error:
## ! object 'epoch' not found
```

``` r
task <- tsk("melanoma")
task
```

```
## 
## ── <TaskClassif> (32701x5): Melanoma Classification ────────────────────────────
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
```

```
## Error in dictionary_sugar_get(dict = dict, .key = .key, ..., .dicts_suggest = .dicts_suggest): Cannot set argument 'weight' for 'TorchLoss' (not a constructor argument, not a parameter, not a field). Did you mean 'class_weight'?
```

``` r
preprocessing <- po("classbalancing", ratio = 4, reference = "minor",
    adjust = "minor") %>>%
  po("augment_random_horizontal_flip") %>>%
  po("augment_random_vertical_flip") %>>%
  po("augment_random_crop", size = c(128, 128), pad_if_needed = TRUE)
glrn <- as_learner(preprocessing %>>% model)
```

```
## Error in concat_graphs(g1, g2, in_place = FALSE): Output type of PipeOp augment_random_crop during training (Task) incompatible with input type of PipeOp torch_model_regr (ModelDescriptor)
```

``` r
library("mlr3viz")
glrn$id <- "multimodal"
rr <- resample(task, glrn, rsmp("cv", folds = 5))
```

```
## Warning: Caught mlr3error. Canceling all iterations ...
```

```
## Error in private$.train(input): No missing values allowed in task 'melanoma'.
## This happened PipeOp torch_ingress_ltnsr's $train()
```

``` r
autoplot(rr, type = "roc")
```

```
## Error in FUN(X[[i]], ...): Need a binary classification problem to plot a ROC curve
```
