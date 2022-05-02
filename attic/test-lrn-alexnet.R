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
  img %>%
    # first convert image to tensor
    torchvision::transform_to_tensor() %>%# $to(device = choose_device()) %>%
    # # then move to the GPU (if available)
    (function(x) x$to(device = choose_device())) %>%
    # Required resize for alexnet
    torchvision::transform_resize(c(64, 64))
}


lrn_alexnet <- lrn("classif.alexnet",
                   predict_type = "prob",
                   num_threads = 15,
                   # Can't use pretrained on 10-class dataset yet, expects 1000
                   pretrained = TRUE,
                   img_transform_train = img_transforms,
                   img_transform_val = img_transforms,
                   img_transform_predict = img_transforms,
                   batch_size = 10,
                   epochs = 15,
                   device = choose_device()
                   )


lrn_alexnet$train(img_task)


# lrn_alexnet$model


img_test <- df_from_imagenet_dir("/opt/example-data/imagenette2-160/val/")

# Make it a task
img_task_test <- mlr3::as_task_classif(img_task_df, target = "target")

preds <- lrn_alexnet$predict(img_task_test)

preds$score(msr("classif.acc"))
preds$score(msr("classif.ce"))
