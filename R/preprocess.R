#' @title Lazy Preprocessing and Transformations
#' @description
#' Overview over all implemented preprocessing methods.
#' See [`PipeOpTaskPreprocTorch`] for more details.
#' @name preprocessing_pipeops
#' @rawNamespace exportPattern("^PipeOpPreproc")
#' @include PipeOpTaskPreprocTorch.R
#' @section Available PipeOps:
#' The following
NULL

#' @name PipeOpPreprocResize
#' @describeIn preprocessing_pipeops See [`torchvision::transform_resize`]
#' @rdname preprocessing_pipeops
#' @section preproc_resize:
#' Resize a tensor to a given size
register_preproc("resize", torchvision::transform_resize,
  packages = "torchvision",
  param_set = ps(
    size = p_uty(tags = c("train", "required")),
    interpolation = p_fct(levels = magick::filter_types(), special_vals = list(0L, 2L, 3L),
      tags = "train", default = 2L
    )
  ),
  shapes_out = function(shapes_in, param_vals, task) {
    size = param_vals$size
    shape = shapes_in[[1L]]
    assert_true(length(shape) > 2)
    height = shape[[length(shape) - 1L]]
    width = shape[[length(shape)]]
    s = torchvision::transform_resize(torch_ones(c(1, height, width), device = "meta"), size = size)$shape[2:3]
    list(c(shape[seq_len(length(shape) - 2L)], s))
  }
)
