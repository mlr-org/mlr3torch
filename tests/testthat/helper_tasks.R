load_task_nano_imagenet = function(id = "nano_imagenet") {
  path = testthat::test_path("assets", "nano_imagenet")
  images = list.files(file.path(path, "images"))
  images = imageuri(paste0(path, "/images/", images))
  labels = readRDS(file.path(path, "labels.rds"))
  d = data.table(
    class = labels,
    image = images
  )

  as_task_classif(d, target = "class", id = id)
}

#' @include zzz.R
mlr3::mlr_tasks$add("nano_imagenet", load_task_nano_imagenet)
