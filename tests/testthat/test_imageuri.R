# test_that("imageuri, works", {
#   # TODO: need some random pictures for that in inst
#   uris = list.files(system.file("inst", "toytask", "images", package = "mlr3torch"), full.name = TRUE)
#   images = imageuri(uris)
#   expect_true(inherits(images, "imageuri"))
# })
#
# test_that("Subsetting of data works", {
#   task = tsk("tiny_imagenet")
#   imagecol = task$data()[["image"]]
#   print(imagecol)
#   p = po("imagetrafo", .trafo = "vflip")
#   expect_true(inherits(imagecol[2:5], "imageuri"))
# })
#
