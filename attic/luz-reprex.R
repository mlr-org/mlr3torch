# From https://torchvision.mlverse.org/articles/examples/tinyimagenet-alexnet.html

# Packages ----------------------------------------------------------------

library(torch)
library(torchvision)

# Datasets ----------------------------------------------------------------

dir <- "/opt/example-data/tiny-imagenet"

device <- if(cuda_is_available()) "cuda" else "cpu"

to_device <- function(x, device) {
  x$to(device = device)
}

train_ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  transform = function(x) {
    x %>%
      transform_to_tensor() %>%
      to_device(device) %>%
      transform_resize(c(64, 64))
  }
)

valid_ds <- tiny_imagenet_dataset(
  dir,
  download = TRUE,
  split = "val",
  transform = function(x) {
    x %>%
      transform_to_tensor() %>%
      to_device(device) %>%
      transform_resize(c(64, 64))
  }
)

train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)

# Luzify ------------------------------------------------------------------
library(torch)
library(torchvision)
library(luz)

# Getting a pretrained model
model <- model_alexnet(pretrained = FALSE, num_classes = 10)

model %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam
  )

# Defining alexnet manually:
# https://github.com/mlverse/torchvision/blob/main/R/models-alexnet.R#L2-L37
# same as calling torchvision:::alexnet
alexnet <- torch::nn_module(
  "AlexNet",
  initialize = function(num_classes = 1000) {
    self$features <- torch::nn_sequential(
      torch::nn_conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2),
      torch::nn_conv2d(64, 192, kernel_size = 5, padding = 2),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2),
      torch::nn_conv2d(192, 384, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(384, 256, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_conv2d(256, 256, kernel_size = 3, padding = 1),
      torch::nn_relu(inplace = TRUE),
      torch::nn_max_pool2d(kernel_size = 3, stride = 2)
    )
    self$avgpool <- torch::nn_adaptive_avg_pool2d(c(6,6))
    self$classifier <- torch::nn_sequential(
      torch::nn_dropout(),
      torch::nn_linear(256 * 6 * 6, 4096),
      torch::nn_relu(inplace = TRUE),
      torch::nn_dropout(),
      torch::nn_linear(4096, 4096),
      torch::nn_relu(inplace = TRUE),
      torch::nn_linear(4096, num_classes)
    )
  },
  forward = function(x) {
    x <- self$features(x)
    x <- self$avgpool(x)
    x <- torch_flatten(x, start_dim = 2)
    x <- self$classifier(x)
  }
)

alexnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam
  )

# Why is model_alexnet() missing a class?
class(alexnet)
class(torchvision:::alexnet)
class(torchvision::model_alexnet(pretrained = FALSE))
class(torchvision::model_alexnet(pretrained = TRUE))

# Session info ------------------------------------------------------------
sessioninfo::session_info(pkgs = c("torch", "torchvision", "luz"))


