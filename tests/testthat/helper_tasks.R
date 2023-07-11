nano_cats_vs_dogs = function(id = "nano_cats_vs_dogs", image_type = "imageuri_vector") {
  assert_string(id)
  assert_choice(image_type, c("imageuri_vector", "image_vector"))
  path = testthat::test_path("assets", "nano_cats_vs_dogs")
  image_names = list.files(path)
  uris = normalizePath(file.path(path, image_names))

  if (image_type == "image_vector") {
    images = lapply(uris, function(uri) {
      as_array(torchvision::transform_to_tensor(magick::image_read(uri)))

    })
    images = image_vector(images)
  } else {
    images = imageuri_vector(uris)
  }

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

nano_mnist = function(id = "nano_mnist", image_type = "image_vector") {
  assert_string(id)
  assert_choice(image_type, c("imageuri_vector", "image_vector"))
  path = testthat::test_path("assets", "nano_mnist")
  image_names = list.files(path)
  uris = normalizePath(file.path(path, image_names))

  if (image_type == "image_vector") {
    images = lapply(uris, function(uri) {
      as_array(torchvision::transform_to_tensor(magick::image_read(uri)))
    })
    images = image_vector(images)
  } else {
    images = imageuri_vector(uris)
  }

  labels = map_chr(image_names, function(name) {strsplit(name, split = "")[[1L]][1L]})

  dat = data.table(x = images, letter = labels)

  task = as_task_classif(dat, id = "nano_mnist", label = "Letter Classification", target = "letter")
  task
}
