# #' @title Tiny Imagenet Classification Task
# #' It only containst the training data of the task.
# #' @format [R6::R6Class] inheriting from [TaskClassif].
#
# #' @name mlr_tasks_iris
# #'
# NULL
#
# load_task_tiny_imagenet = function(id = "tiny_imagenet", download = FALSE) {
#   rlang::local_options(timeout = 300L) # download takes long
#   cache_dir = tools::R_user_dir("mlr3torch", "cache")
#   superdir = file.path(cache_dir, "tiny-imagenet-200")
#   if (download) {
#     torchvision::tiny_imagenet_dataset(root = superdir, download = TRUE)
#   } else {
#     assert_directory_exists(superdir)
#   }
#
#   lookup = fread(sprintf("%s/words.txt", superdir), header = FALSE)
#   colnames(lookup) = c("id", "label")
#
#   train_dir = sprintf("%s/train", superdir)
#   train_ids = list.files(train_dir)
#   train_folders = sprintf("%s/%s", train_dir, train_ids)
#   train_uris = map(train_folders,
#     function(train_folder) {
#       image_dir = sprintf("%s/images", train_folder)
#       sprintf("%s/%s", image_dir, list.files(image_dir))
#     }
#   )
#
#   train_labels = lookup[train_ids, "label", on = "id"][[1L]]
#   train_labels = rep(train_labels, times = lengths(train_uris))
#   train_labels = factor(train_labels)
#   train_uris = unlist(train_uris)
#
#   images = imageuri(train_uris)
#   d = data.table(
#     class = train_labels,
#     image = images
#   )
#
#   as_task_classif(d, target = "class", id = id)
# }
#
# #' @include zzz.R
# mlr3torch_image_tasks[["tiny_imagenet"]] = load_task_tiny_imagenet
