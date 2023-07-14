nano_cats_vs_dogs = function(id = "nano_dogs_vs_cats") {
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
