library(mlr3)
library(mlr3torch)
library(torch)

device <- if(cuda_is_available()) "cuda" else "cpu"
to_device <- function(x, device) x$to(device = device)

img_task_df <- df_from_imagenet_dir(c(
  "/opt/example-data/imagenette2-160/train/"
  # "/opt/example-data/imagenette2-160/val/"
))

# Subsample for faster testing, 5 imgs per class
img_task_df <- img_task_df[img_task_df[, .I[sample.int(.N, min(min(5L, .N), .N))], by = .(target)]$V1]

# Make it a task
img_task <- mlr3::as_task_classif(img_task_df, target = "target")

img_transforms <- function(img) {
  img %>%
    # first convert image to tensor
    torchvision::transform_to_tensor() %>%
    # # then move to the GPU (if available)
    (function(x) x$to(device = device)) %>%
    # Required resize for alexnet
    torchvision::transform_resize(c(64,64))
}


lrn_alexnet <- lrn("classif.torch.alexnet",
                   predict_type = "response",
                   num_threads = 15,
                   # Can't use pretrained on 10-class dataset yet, expects 1000
                   pretrained = FALSE,
                   img_transforms = img_transforms,
                   batch_size = 10,
                   epochs = 2,
                   device = device
                   )


lrn_alexnet$train(img_task)


# lrn_alexnet$model

lrn_alexnet$predict(img_task, row_ids = 1:10)


# predict method experiments ----------------------------------------------

test_ds <- img_dataset(img_task$data(), row_ids = 1:10, transform = img_transforms)
test_dl <- torch::dataloader(test_ds, batch_size = 1, shuffle = TRUE, drop_last = FALSE)

lrn_alexnet$model$eval()

pred_prob <- numeric(0)
pred_class <- integer(0)

torch::with_no_grad({
  coro::loop(for (b in test_dl) {
    pred <- lrn_alexnet$model(b[[1]]$to(device = "cpu"))

    pred_prob <- c(pred_prob, pred$softmax(dim = 2))

    pred <- as.integer(pred$argmax())
    pred_class <- c(pred_class, pred)
  })
})

# Numeric class prediction to class label somehow
targets <- img_task$data(cols = "target")[[1]]
levels(targets)[pred_class]

# Tabularize class predictions.. but not as wonky?

lapply(pred_prob, function(x) {
  x |>
    as.numeric() |>
    t() |>
    as.data.frame()
}) |>
  data.table::rbindlist()
