load_task_nano_imagenet = function(id = "nano_imagenet") {
  path = system.file("toytask", package = "mlr3torch")
  images = list.files(sprintf("%s/images", path))
  images = imageuri(paste0(path, "/images/", images))
  labels = readRDS(paste0(path, "/labels.rds"))
  d = data.table(
    class = labels,
    image = images
  )

  as_task_classif(d, target = "class", id = id)
}

register_task("nano_imagenet", load_task_nano_imagenet)
