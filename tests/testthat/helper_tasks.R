nano_cats_vs_dogs = function(id = "nano_cats_vs_dogs") {
  assert_string(id)
  path = testthat::test_path("assets", "nano_cats_vs_dogs")
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

  dat = data.table(x = images, anmimal = labels)

  task = as_task_classif(dat, id = "nano_cats_vs_dogs", label = "Cats vs Dogs", target = "animal", positive = "cat")
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
