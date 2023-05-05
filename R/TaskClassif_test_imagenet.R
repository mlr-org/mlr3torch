# #' @title Mini Imagenet Classification Task
# #' It only containst the training data of the task.
# #' @format [R6::R6Class] inheriting from [TaskClassif].
# #' @name mlr_tasks_iris
# #'
# NULL

load_task_test_imagenet = function(id = "test_imagenet") {
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
