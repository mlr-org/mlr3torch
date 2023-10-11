nano_dogs_vs_cats = function(id = "nano_dogs_vs_cats") {
  assert_string(id)
  path = testthat::test_path("assets", "nano_dogs_vs_cats")
  image_names = list.files(path)
  uris = normalizePath(file.path(path, image_names))

  images = imageuri(uris)

  labels = map_chr(image_names, function(name) {
    if (startsWith(name, "cat")) {
      "cat"
    } else if (startsWith(name, "dog")) {
      "dog"
    } else {
      stopf("Invalid image name %s, name.", name)
    }
  })

  labels = factor(labels)

  dat = data.table(x = images, animal = labels)

  task = as_task_classif(dat, id = "nano_dogs_vs_cats", label = "Cats vs Dogs", target = "animal", positive = "cat")
  task
}

nano_mnist = function(id = "nano_mnist") {
  assert_string(id)
  path = testthat::test_path("assets", "nano_mnist")
  data = readRDS(file.path(path, "data.rds"))

  ds = dataset(
    initialize = function(images) {
      self$images = torch_tensor(images, dtype = torch_float32())
    },
    .getbatch = function(idx) {
      list(image = self$images[idx, , , drop = FALSE])
    },
    .length = function() dim(self$images)[1L]
  )(data$image)

  data_descriptor = DataDescriptor(dataset = ds, list(image = c(NA, 1, 28, 28)))

  dt = data.table(
    image = lazy_tensor(data_descriptor),
    label = droplevels(data$label),
    row_id = seq_along(data$label)
  )

  backend = DataBackendDataTable$new(data = dt, primary_key = "row_id")

  task = TaskClassif$new(
    backend = backend,
    id = "nano_mnisst",
    target = "label",
    label = "MNIST Nano"
  )

  task$col_roles$feature = "image"

  task
}

nano_imagenet = function(id = "nano_imagenet") {
  assert_string(id)

  path = testthat::test_path("assets", "nano_imagenet")
  image_names = list.files(path)
  uris = normalizePath(file.path(path, image_names))

  images = imageuri(uris)

  labels = map_chr(image_names, function(name) {strsplit(name, split = "_")[[1L]][1L]})

  dat = data.table(image = images, class = labels)

  task = as_task_classif(dat, id = "nano_imagenet", label = "Nano Imagenet", target = "class")
  task
}
