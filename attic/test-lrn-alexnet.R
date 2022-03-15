library(mlr3)
library(mlr3torch)
library(torch)

# to_device <- function(x, device) x$to(device = device)

img_task_df <- df_from_imagenet_dir("/opt/example-data/imagenette2-160/train/")

# Subsample for faster testing, 5 imgs per class
img_task_df <- img_task_df[img_task_df[, .I[sample.int(.N, min(min(5L, .N), .N))], by = .(target)]$V1]

# Make it a task
img_task <- mlr3::as_task_classif(img_task_df, target = "target")

img_transforms <- function(img) {
  # Transformation to tensor is always required
  img <- torchvision::transform_to_tensor(img)
  # Move to GPU if available
  img$to(device = choose_device())
  # Resize images
  img <- torchvision::transform_resize(img, size = c(64, 64))

  img
}


lrn_alexnet <- lrn("classif.torch.alexnet",
                   predict_type = "response",
                   num_threads = 15,
                   valid_split = 0.2,
                   pretrained = TRUE,
                   img_transform_train = img_transforms,
                   img_transform_predict = img_transforms,
                   batch_size = 10,
                   learn_rate = 0.03,
                   epochs = 5,
                   device = choose_device()
                   )

lrn_alexnet$param_set

lrn_alexnet$train(img_task)

# Luz also gives us metrics for free
names(lrn_alexnet$model)
data.table::rbindlist(lrn_alexnet$model$records$metrics$train)


img_test <- df_from_imagenet_dir("/opt/example-data/imagenette2-160/val/")

# Make it a task
img_task_test <- mlr3::as_task_classif(img_test, target = "target")

preds <- lrn_alexnet$predict(img_task_test)

preds$score(msr("classif.acc"))
preds$score(msr("classif.ce"))
