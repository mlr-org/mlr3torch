test_that("PipeOpImageTrafo works", {
  task = tsk("tiny_imagenet")
  batch = get_batch(task, 1L, device = "cpu")
  graph = po("imagetrafo", .trafo = "to_tensor") %>>%
    po("imagetrafo", .trafo = "vflip")

  out = graph$train(task)
  task_vflip = out$vflip.output
  batch_vflip = get_batch(task_vflip, batch_size = 1L, device = "cpu")
  img = batch$x
  img_vflip = batch_vflip$x
  expect_true(torch_equal(torchvision::transform_vflip(img), img_vflip))
})
