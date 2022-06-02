#' @title Image Transformation
#' @description
#' image transformation
#' @export
PipeOpImageTrafo = R6Class("TorchOpImageTrafo",
  inherit = mlr3pipelines::PipeOpTaskPreprocSimple,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param .trafo (`character(1)`)\cr
    #'   The transformation to apply.
    initialize = function(id = .trafo, param_vals = list(), .trafo) {
      assert_choice(.trafo, torch_reflections$image_trafos)
      private$.trafo = .trafo
      param_set = paramsets_image_trafo$get(.trafo)
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals
      )
    }
  ),
  private = list(
    .train_task = function(task) {
      pars = self$param_set$get_values(tags = "train")
      .data = task$backend$.__enclos_env__$private$.data
      image_cols = colnames(.data)[map_lgl(.data, function(x) inherits(x, "imageuri"))]
      torch_trafo = get_image_trafo(private$.trafo)
      trafo = function(img) {
        invoke(torch_trafo, img = img, .args = pars)
      }
      for (image_col in image_cols) {
        .data[, (image_col) := transform_imageuri(get(..image_col), ..trafo)]
      }
      task$backend$.__enclos_env__$private$.data = .data
      task
    },
    .predict_task = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      task
    },
    .trafo = NULL
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("imagetrafo", value = function() stopf("Please use po('imagetrafo', ...))."))
