# Adapted from https://torchvision.mlverse.org/articles/examples/tinyimagenet-alexnet.html

# Packages ----------------------------------------------------------------

library(torch)
library(torchvision)
library(mlr3torch)

# Datasets ----------------------------------------------------------------

# dir <- "/opt/example-data/tiny-imagenet"
#
device <- if(cuda_is_available()) "cuda" else "cpu"
#
# to_device <- function(x, device) {
#   x$to(device = device)
# }
#
# train_ds <- tiny_imagenet_dataset(
#   "/opt/example-data/tiny-imagenet",
#   download = TRUE,
#   transform = function(x) {
#     x %>%
#       transform_to_tensor() %>%
#       #to_device(device) %>%
#       transform_resize(c(64, 64))
#   }
# )
#
# valid_ds <- tiny_imagenet_dataset(
#   "/opt/example-data/tiny-imagenet",
#   download = TRUE,
#   split = "val",
#   transform = function(x) {
#     x %>%
#       transform_to_tensor() %>%
#       to_device(device) %>%
#       transform_resize(c(64,64))
#   }
# )

# Custom dataset from task ------------------------------------------------
img_task_df <- df_from_imagenet_dir(c(
  "/opt/example-data/imagenette2-160/train/",
  "/opt/example-data/imagenette2-160/val/"
))

# img_task <- mlr3::as_task_classif(img_task_df, target = "target")

train_idx <- sample(nrow(img_task_df), 256)
val_idx <- sample(setdiff(seq_len(nrow(img_task_df)), train_idx), 64)

img_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # # then move to the GPU (if available)
    # (function(x) x$to(device = device)) %>%
    # Required resize for alexnet
    transform_resize(c(64,64))
}

train_ds <- img_dataset(img_task_df, row_ids = train_idx, transform = img_transforms)
valid_ds <- img_dataset(img_task_df, row_ids = val_idx, transform = img_transforms)

train_ds$.length()
train_ds$num_classes

valid_ds$.length()
valid_ds$num_classes

# Back to original code

train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE, drop_last = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 32, shuffle = FALSE, drop_last = TRUE)


# Model -------------------------------------------------------------------

model <- model_alexnet(pretrained = FALSE, num_classes = train_ds$num_classes)
# model$to(device = device)

optimizer <- optim_adam(model$parameters)
scheduler <- lr_step(optimizer, step_size = 1, 0.95)
loss_fn <- nn_cross_entropy_loss()


# Training loop -----------------------------------------------------------

train_step <- function(batch) {
  optimizer$zero_grad()
  output <- model(batch[[1]]$to(device = device))
  loss <- loss_fn(output, batch[[2]]$to(device = device))
  loss$backward()
  optimizer$step()
  loss
}

valid_step <- function(batch) {
  # browser()
  model$eval()
  pred <- model(batch[[1]]$to(device = device))
  pred <- torch_topk(pred, k = 5, dim = 2, TRUE, TRUE)[[2]]
  pred <- pred$to(device = torch_device("cpu"))
  correct <- batch[[2]]$view(c(-1, 1))$eq(pred)$any(dim = 2)
  model$train()
  correct$to(dtype = torch_float32())$mean()$item()
}

for (epoch in 1:2) {

  pb <- progress::progress_bar$new(
    total = length(train_dl),
    format = "[:bar] :eta Loss: :loss"
  )

  l <- c()
  coro::loop(for (b in train_dl) {
    loss <- train_step(b)
    l <- c(l, loss$item())
    pb$tick(tokens = list(loss = mean(l)))
  })

  acc <- c()
  with_no_grad({
    coro::loop(for (b in valid_dl) {
      accuracy <- valid_step(b)
      acc <- c(acc, accuracy)
    })
  })

  scheduler$step()
  cat(sprintf("[epoch %d]: Loss = %3f, Acc= %3f \n", epoch, mean(l), mean(acc)))
}


# Predict -----------------------------------------------------------------

model$eval()

test_img <- train_ds$.getitem(1)$x
test_label <- train_ds$.getitem(1)$y
dim(test_img)

test_img <- test_img$unsqueeze(1)
dim(test_img)


pred <- model(test_img)

dim(pred)

pred_k <- torch_topk(pred, k = 10, dim = 2, TRUE, TRUE)[[2]]


# Luzify ------------------------------------------------------------------
# Not working for either luz- or torchvision reasons.
library(luz)

model %>%
  setup(
     loss = nn_cross_entropy_loss(),
     optimizer = optim_adam
  ) %>%
  fit(train_dl, epochs = 2, valid_data = valid_dl)
