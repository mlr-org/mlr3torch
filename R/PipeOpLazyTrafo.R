#' @title Apply a Transformation to a Tensor
#' @description
#' Apply a transformation to tensor(like) objects.
#'
#' @export
PipeOpTrafo = R6Class("PipeOpTrafo",
  inherit = PipeOp,
  public = list(
    operator = NULL,
    initialize = function(id = "trafo", operator, inname = "input", param_vals = list(), packages = character(0)) {
      self$operator = assert_function(operator)

      input = data.table(name = inname, train = "*", predict = "NULL")
      output = data.table(name = "output", train = "*", predict = "NULL")

      super$initialize(
        id = id,
        input = input,
        output = output,
        param_vals = param_vals,
        packages = packages
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      self$state = list()
      do.call(self$operator, unnames(inputs))
    },
    .predict = function(inputs) {
      list(NULL)
    }
  )
)

PipeOpLazyTrafo = R6Class("PipeOpLazyTrafo",
  inherit = PipeOpTaskPreprocSimple,
  public = list(
    initialize = function(id, param_vals = list(), param_set = ps(), packages = "torch", tags = NULL) {
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        packages = c("mlr3torch", "mlr3pipelines"),
        feature_types = "lazy_tensor"
      )
    }
  ),
  private = list(
    .transform = function(task) {
      cols = self$state$dt_columns
      if (!length(cols)) {
        return(task)
      }
      dt = task$data(cols = cols)
      pars = self$param_set$get_values()

      # now we need to apply the transformation to every element in all columns
      # 1. To every

      data_descriptors = new.env()

      walk(dt, function(col) {
        walk(col, function(x) {
          if (!exists(x$.hash, data_descriptors)) {
            data_descriptors[[x$.hash]] = x
          }
        })
      })

      addresses = map(dt, function(col) {
        map_chr(col, data.table::address)
      })

      data_descriptors = unique(unlist(addresses))

      if (!drop) {
        features = cbind(dt, features)
      }

      task$select(setdiff(task$feature_names, cols))$cbind(features)
      return(task)
    },
    .apply = function(cols) {

    }
  )
)

#' @title Compare two images
#' @description
#' Calls [`magick::image_compare()`] on two columns and crates a new column.
#' @export
PipeOpImageCompare = R6Class("PipeOpImageCompare",
  inherit = PipeOpLazyTrafo,
  public = list(
    initialize = function(id = "image_compare", param_vals = list()) {
      param_set = ps(
        cols = p_uty(tags = c("train", "required"))
      )
    }
  )
)
