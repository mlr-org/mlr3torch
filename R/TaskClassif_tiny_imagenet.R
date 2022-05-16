# #' @title Tiny Imagenet Classification Task
# #' It only containst the training data of the task.
# #' @format [R6::R6Class] inheriting from [TaskClassif].
# #' @name mlr_tasks_iris
# #' @include mlr_tasks.R
# #'
# NULL

load_task_tiny_imagenet = function(id = "tiny_imagenet") {
  rlang::local_options(timeout = 120L) # download takes long
  cache_dir = R_user_dir("mlr3torch", "cache")
  superdir = sprintf("%s/data", cache_dir)
  dir = sprintf("%s/tiny-imagenet-200", superdir)

  torchvision::tiny_imagenet_dataset(root = superdir, download = TRUE)

  lookup = fread(sprintf("%s/words.txt", dir), header = FALSE)
  colnames(lookup) = c("id", "label")

  train_dir = sprintf("%s/train", dir)
  train_ids = list.files(train_dir)
  train_folders = sprintf("%s/%s", train_dir, train_ids)
  train_uris = map(train_folders,
    function(train_folder) {
      image_dir = sprintf("%s/images", train_folder)
      sprintf("%s/%s", image_dir, list.files(image_dir))
    }
  )

  train_labels = lookup[train_ids, "label", on = "id"][[1L]]
  train_labels = rep(train_labels, times = lengths(train_uris))
  train_labels = factor(train_labels)
  train_uris = unlist(train_uris)

  images = imageuri(train_uris)
  d = data.table(
    class = train_labels,
    image = images
  )

  task = as_task_classif(d, target = "class", id = id)
}
