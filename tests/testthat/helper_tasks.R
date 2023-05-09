nano_imagenet = function(id = "nano_imagenet") {
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
