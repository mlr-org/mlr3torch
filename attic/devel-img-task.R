library(torch)
library(torchvision)
library(luz)
library(mlr3torch)



imgnette <- df_from_imagenet_dir(c(
     "/opt/example-data/imagenette2-160/train/",
     "/opt/example-data/imagenette2-160/val/"
   ))

# create dataset from image folder
imagenette160_ds <- imgnet_dataset(imgnette, path = "image", class = "class")

# test if default methods work as expected
imagenette160_ds$.length()
imagenette160_ds$.nclass()
imagenette160_ds$.getitem(1)




# Train -------------------------------------------------------------------

# Small subsample for now, just testing basic functionality
# imgnette is sorted by class so have to sample randomly
train_idx <- sample(nrow(imgnette), 128)
val_idx <- sample(setdiff(seq_len(nrow(imgnette)), train_idx), 32)


img_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # # then move to the GPU (if available)
    # (function(x) x$to(device = device)) %>%
    # data augmentation
    transform_random_resized_crop(size = c(160, 160))
}


imagenette160_ds_train <- imgnet_dataset(imgnette[train_idx, ], transform = img_transforms)
imagenette160_ds_val <- imgnet_dataset(imgnette[val_idx, ], transform = img_transforms)

imagenette160_train_dl <- dataloader(imagenette160_ds_train, batch_size = 32)
imagenette160_val_dl <- dataloader(imagenette160_ds_val, batch_size = 32)

net <- nn_module(
  "Net",
  initialize = function() {
    self$conv1 <- nn_conv2d(3, 32, 3, 1)
    self$conv2 <- nn_conv2d(32, 64, 3, 1)
    self$dropout1 <- nn_dropout2d(0.25)
    self$dropout2 <- nn_dropout2d(0.5)
    self$fc1 <- nn_linear(9216, 128)
    self$fc2 <- nn_linear(128, 10)
  },
  forward = function(x) {
    x %>%
      self$conv1() %>%
      nnf_relu() %>%
      self$conv2() %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      self$dropout1() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$dropout2() %>%
      self$fc2()
  }
)


fitted <- net %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>%
  fit(imagenette160_train_dl, epochs = 5, valid_data = imagenette160_val_dl)

# Making predictions ------------------------------------------------------

preds <- predict(fitted, test_dl)
preds$shape




classes <- imgnette$class


# torchvision way ---------------------------------------------------------
library(torch)
library(torchvision)
library(luz)

img_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # # then move to the GPU (if available)
    # (function(x) x$to(device = device)) %>%
    # data augmentation
    transform_resize(size = c(160, 160))
}

tv_dataset_train <- image_folder_dataset("/opt/example-data/imagenette2-160/train/", loader = magick_loader, transform = img_transforms)
tv_dataset_val <- image_folder_dataset("/opt/example-data/imagenette2-160/val/", loader = magick_loader, transform = img_transforms)

tv_train_dl <- dataloader(tv_dataset_train, batch_size = 32, shuffle = TRUE)
tv_valid_dl <- dataloader(tv_dataset_val, batch_size = 32)


# VGG ---------------------------------------------------------------------
vgg11 <- model_vgg11(pretrained = FALSE, num_classes = 10)

# Adapt output feature count
# last_layer_in <- vgg11$classifier$`6`$in_features
# vgg11$classifier$`6`$out_feature
#
# vgg11$classifier$`6` <- nn_linear(in_features = last_layer_in, out_features = length(tv_dataset_train$classes))

fitted <- vgg11 %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>%
  set_hparams(num_class = 10) %>%
  set_opt_hparams(lr = 0.003) %>%
  fit(tv_train_dl, epochs = 5, valid_data = tv_valid_dl)


# Resnet ------------------------------------------------------------------
resnet <- model_resnet18(pretrained = TRUE)

device <- if (cuda_is_available()) torch_device("cuda:0") else "cpu"

train_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    transform_to_tensor() %>%
    # then move to the GPU (if available)
    (function(x) x$to(device = device)) %>%
    # data augmentation
    transform_random_resized_crop(size = c(224, 224)) %>%
    # data augmentation
    transform_color_jitter() %>%
    # data augmentation
    transform_random_horizontal_flip() %>%
    # normalize according to what is expected by resnet
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

valid_transforms <- function(img) {
  img %>%
    transform_to_tensor() %>%
    (function(x) x$to(device = device)) %>%
    transform_resize(256) %>%
    transform_center_crop(224) %>%
    transform_normalize(mean = c(0.485, 0.456, 0.406), std = c(0.229, 0.224, 0.225))
}

tv_dataset_train <- image_folder_dataset("/opt/example-data/imagenette2-160/train/", loader = magick_loader, transform = train_transforms)
tv_dataset_val <- image_folder_dataset("/opt/example-data/imagenette2-160/val/", loader = magick_loader, transform = valid_transforms)

tv_train_dl <- dataloader(tv_dataset_train, batch_size = 32, shuffle = TRUE)
tv_valid_dl <- dataloader(tv_dataset_val, batch_size = 32)

num_features <- resnet$fc$in_features

resnet$fc <- nn_linear(in_features = num_features, out_features = length(tv_dataset_train$classes))

fitted <- resnet %>%
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) %>%
  set_hparams(num_class = 10) %>%
  set_opt_hparams(lr = 0.003) %>%
  fit(tv_train_dl, epochs = 5, valid_data = tv_valid_dl)


criterion <- nn_cross_entropy_loss()

optimizer <- optim_sgd(resnet$parameters, lr = 0.05, momentum = 0.9)
num_epochs = 10

scheduler <- optimizer %>%
  lr_one_cycle(max_lr = 0.05, epochs = num_epochs, steps_per_epoch = tv_train_dl$.length())


train_batch <- function(b) {

  optimizer$zero_grad()
  output <- resnet(b[[1]])
  loss <- criterion(output, b[[2]]$to(device = device))
  loss$backward()
  optimizer$step()
  scheduler$step()
  loss$item()

}

valid_batch <- function(b) {

  output <- resnet(b[[1]])
  loss <- criterion(output, b[[2]]$to(device = "cpu"))
  loss$item()
}

for (epoch in 1:num_epochs) {

  resnet$train()
  train_losses <- c()

  coro::loop(for (b in tv_train_dl) {
    loss <- train_batch(b)
    train_losses <- c(train_losses, loss)
  })

  resnet$eval()
  valid_losses <- c()

  coro::loop(for (b in tv_valid_dl) {
    loss <- valid_batch(b)
    valid_losses <- c(valid_losses, loss)
  })

  cat(sprintf("\nLoss at epoch %d: training: %3f, validation: %3f\n", epoch, mean(train_losses), mean(valid_losses)))
}
