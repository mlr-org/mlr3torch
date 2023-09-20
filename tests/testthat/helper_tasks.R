nano_dogs_vs_cats = function(id = "nano_dogs_vs_cats", size = c(64, 64)) {
  assert_string(id)
  path = testthat::test_path("assets", "nano_dogs_vs_cats")
  image_names = list.files(path)
  uris = normalizePath(file.path(path, image_names))

  ds = dataset(
    initialize = function(pths) {
      self$pths = pths
    },
    .getitem = function(id) {
      x = magick::image_read(self$pths[id])
      list(image = torchvision::transform_resize(x, size))
    },
    .length = function() {
      length(self$pths)
    }
  )(uris)

  dd = DataDescriptor(ds, list(image = c(3, size)))

  images = lazy_tensor(dd)

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
  image_names = list.files(path)
  uris = normalizePath(file.path(path, image_names))

  images = imageuri(uris)

  labels = map_chr(image_names, function(name) {strsplit(name, split = "")[[1L]][1L]})

  dat = data.table(x = images, letter = labels)

  task = as_task_classif(dat, id = "nano_mnist", label = "Letter Classification", target = "letter")
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

nano_random_tensor = function(id = "random_tensor") { # nolint
  ds = dataset(
    name = id,
    initialize = function() {
      self$tensor = torch_randn(100, 5, 10)
    },
    .getitem = function(id) {
      self$tensor[id,]

    },
    .length = function() {
      nrow(self$tensor)
    }
  )
  lt = lazy_tensor(ds(), c(5, 10))
  y = rnorm(10)

  dt = data.table(y = y, x = lt)
  task = as_task_regr(dt, id = id, label = "Random Tensor", target = "y")


}
