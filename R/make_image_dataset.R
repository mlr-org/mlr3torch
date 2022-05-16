make_image_dataset = function(task, augmentation = NULL) {
  y = task$data(cols = task$target_names)[[1L]]
  y = as.integer(y)
  y = torch_tensor(y, dtype = torch_long())
  uri = task$data(cols = task$feature_names)[[1L]]
  trafos = c(attr(uri, "trafos"), augmentation)
  image = attr(uri, "transformed")
  assert_true(inherits(image, "magick-image") || inherits(image, "torch_tensor"))
  if (inherits(image, "magick-image")) {
    trafos = c(trafos, torchvision::transform_to_tensor)

  }

  dataset(
    name = task$id,
    initialize = function(y, uri, trafos) {
      self$trafos = trafos
      self$y = y
      self$uri = uri
    },
    # Here we implement .getitem because this probably makes parallelization easier (?)
    .getitem = function(index) {
      y = self$y[index, drop = FALSE]
      uri = self$uri[index]
      image = magick::image_read(uri)
      for (trafo in self$trafos) {
        image = trafo(image)
      }
      list(
        y = y,
        x = image
      )
    },
    .length = function() {
      length(self$y)
    }
  )(y, uri, trafos)
}

