# Only used for testing
toytask = function(id = "toytask") {
  path = system.file("toytask", package = "mlr3torch")
  images = list.files(sprintf("%s/images", path))
  images = imageuri(paste0(path, "/images/", images))
  labels = readRDS(paste0(path, "/labels.rds"))
  d = data.table(
    class = labels,
    image = images
  )

  task = as_task_classif(d, target = "class", id = id)
}
