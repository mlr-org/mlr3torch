test_that("imageuri_vector works", {
  cls = c("imageuri_vector", "character")
  path = testthat::test_path("assets", "nano_mnist")
  image_names = list.files(path)
  uris = normalizePath(file.path(path, image_names))

  img1 = uris[[1]]
  img2 = uris[[2]]

  images = imageuri_vector(uris)

  # class is correct
  expect_class(images, cls)

  # object is invariant under the constructor
  expect_equal(imageuri_vector(images), images)

  # subsetting works
  expect_equal(imageuri_vector(img1), images[[1]])
  expect_equal(imageuri_vector(img2), images[[2]])
  expect_class(images[1:2], cls)
  expect_equal(images[1], images[[1]])
  expect_equal(images[2], images[[2]])

  # assignment works
  images2 = images
  images3 = images
  images2[1] = img2
  images3[[1]] = img2
  expect_equal(images2, images3)
  expect_equal(images2[1], imageuri_vector(img2))
  expect_equal(images3[1], imageuri_vector(img2))

  expect_error({images[1] = 1}, regexp = "Must be of type 'character'", fixed = TRUE)

  # c() works
  expect_class(c(images[1], images[2]), cls)
  expect_equal(c(images[1], as.character(images[2])), images[1:2])

  expect_error(c(images[1], 1), regexp = "all objects must inherit from 'character'", fixed = TRUE)

  # data.table sanity check
  dt = data.table(image = images)
  expect_class(dt$image, cls)
  expect_equal(dt[1, "image"][[1]], images[1])

  # rbind works with two data.tables
  expect_class(rbind(dt, dt)$image, cls)
})
