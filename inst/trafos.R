devtools::load_all("~/mlr/mlr3torch")

task = tsk("tiny_imagenet")

ds = as_dataset(task)

img = ds$.getbatch(1)$x
torchvision::transform_adjust_brightness(img, brightness_factor = 0.2)
ns = getNamespace("torchvision")
exports = names(torchvision:::.__NAMESPACE__.$exports)
trafo_names = Filter(function(x) startsWith(x, "transform_"), exports)

i = 1

trafo = getFromNamespace(trafo_names[i], "torchvision")

library(torchvision)
# imgt = torchvision::transform_perspective(img)
transform_random_crop(img, size = 10)$shape
transform_center_crop(img, size = 10)$shape
transform_hflip(img)
# transform_ten_crop(img, size = c(10, 10))
transform_adjust_gamma(img, gamma = 0.2)$shape
transform_random_order(img)
transform_adjust_brightness(img)
transform_pad(img, padding = 1)
transform_random_affine(img, degrees = 0.2)
# transform_perspective(img)
transform_affine(img, angle = 0.2, translate = c(0.3, 0.2), scale = 0.3, shear = 0.3)
transform_random_rotation(img, degrees = 0.7, expand = TRUE) # ???
transform_vflip(img)
transform_random_resized_crop(img, size = 7)
transform_crop(img, top = 8, height = 7, left = 1, width = 20)
transform_resized_crop(img, top = 8, height = 7, left = 1, width = 20, size = 2)
# transform_to_tensor(img)
debugonce(transform_random_choice)
transform_random_choice(img, transforms = list(function(x) {x}, function(x) {x + 1L}))
# docu: transforms in torch cannot be a tuple
transform_resize(img, 3)
# transform_random_perspective(img)
transform_rgb_to_grayscale(img[1, ]) # bug: does not work batch-wise
transform_convert_image_dtype(img)
# transform_grayscale(img)
# transform_random_erasing(img)
transform_adjust_saturation(img[1,..], 0.3) # bug: does not work batch-wise
transform_linear_transformation(img, torch_randn(64, 64)) # bug: $view(c(-1, 1))
# transform_five_crop(img)
transform_random_vertical_flip(img) # p has a default but iis not passed
transform_random_horizontal_flip(img, p = 0.3)
transform_color_jitter(img)
transform_adjust_contrast(img[1,], contrast_factor = 2) # does not work batch-wise
transform_rotate(img, angle = 0.7)
# transform_random_grayscale(img)
transform_adjust_hue(img, hue_factor = -0.3) # weird bug
transform_normalize(img, mean = 0.2, std = 0.3)
transform_random_apply(img)



