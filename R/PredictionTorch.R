#' @title Prediction Object for a Generic Torch Task
#'
#' @description
#' The [`Prediction`][mlr3::Prediction] object returned by learners that were trained on a
#' [`TaskTorch`].
#'
#' Because a [`TaskTorch`] can represent very different learning problems, this class does not
#' prescribe how `truth`, `response` and `prob` are stored.
#' Each of them may be an atomic vector, a `matrix()` or a [`data.table`][data.table::data.table],
#' whatever the task's prediction encoder produced; see section *Inference* of [`TaskTorch`] for
#' what the built-in encoders return.
#' The checks that `mlr3` performs are correspondingly weak: it is verified that all elements
#' describe the same number of observations, but not what is in them.
#'
#' @param task ([`TaskTorch`])\cr
#'   The task. Used to extract the default `row_ids` and `truth`.
#' @param row_ids (`integer()`)\cr
#'   The row ids of the predicted observations.
#' @param truth (any)\cr
#'   The ground truth, i.e. what `task$truth()` returned.
#' @param response (any)\cr
#'   The predicted response.
#' @param prob (any)\cr
#'   The predicted probabilities.
#' @param check (`logical(1)`)\cr
#'   Whether to check the consistency of the prediction data.
#'
#' @family Prediction
#' @export
#' @examplesIf torch::torch_is_installed()
#' d = data.frame(x = rnorm(10), y1 = rnorm(10), y2 = rnorm(10))
#' task = as_task_torch(d, target = c("y1", "y2"))
#' PredictionTorch$new(task, response = as.matrix(d[, c("y1", "y2")]))
PredictionTorch = R6Class("PredictionTorch",
  inherit = Prediction,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task = NULL, row_ids = task$row_ids,
      truth = if (!is.null(task)) task$truth(row_ids), response = NULL, prob = NULL, check = TRUE) {
      pdata = discard(list(row_ids = row_ids, truth = truth, response = response, prob = prob), is.null)
      class(pdata) = c("PredictionDataTorch", "PredictionData")
      if (check) {
        pdata = check_prediction_data(pdata)
      }

      self$task_type = "torch_supervised"
      self$man = "mlr3torch::PredictionTorch"
      self$data = pdata
      self$predict_types = intersect(c("response", "prob"), names(pdata))
    }
  ),
  active = list(
    #' @field response (any)\cr
    #'   The predicted response.
    response = function(rhs) {
      assert_ro_binding(rhs)
      self$data$response
    },
    #' @field prob (any)\cr
    #'   The predicted probabilities.
    prob = function(rhs) {
      assert_ro_binding(rhs)
      self$data$prob
    }
  )
)

#' @title Prediction Object for a Generic Unsupervised Torch Task
#'
#' @description
#' The [`Prediction`][mlr3::Prediction] object returned by learners that were trained on a
#' [`TaskTorchUnsupervised`].
#' It is the counterpart of [`PredictionTorch`] for the unsupervised task type and differs from it
#' only in that it has no `truth`: an unsupervised task has no ground truth to compare against, so
#' its measures read whatever they need from the task, see [`msr_torch()`].
#'
#' As for [`PredictionTorch`], `response` and `prob` may be an atomic vector, a `matrix()` or a
#' [`data.table`][data.table::data.table], whatever the task's prediction encoder produced.
#'
#' @param task ([`TaskTorchUnsupervised`])\cr
#'   The task. Used to extract the default `row_ids`.
#' @param row_ids (`integer()`)\cr
#'   The row ids of the predicted observations.
#' @param response (any)\cr
#'   The predicted response.
#' @param prob (any)\cr
#'   The predicted probabilities.
#' @param check (`logical(1)`)\cr
#'   Whether to check the consistency of the prediction data.
#'
#' @family Prediction
#' @export
#' @examplesIf torch::torch_is_installed()
#' d = data.frame(x1 = rnorm(10), x2 = rnorm(10))
#' task = as_task_torch(d, output_dim = 2L)
#' PredictionTorchUnsupervised$new(task, response = as.matrix(d))
PredictionTorchUnsupervised = R6Class("PredictionTorchUnsupervised",
  inherit = Prediction,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task = NULL, row_ids = task$row_ids, response = NULL, prob = NULL,
      check = TRUE) {
      pdata = discard(list(row_ids = row_ids, response = response, prob = prob), is.null)
      class(pdata) = c("PredictionDataTorchUnsupervised", "PredictionData")
      if (check) {
        pdata = check_prediction_data(pdata)
      }

      self$task_type = "torch_unsupervised"
      self$man = "mlr3torch::PredictionTorchUnsupervised"
      self$data = pdata
      self$predict_types = intersect(c("response", "prob"), names(pdata))
    }
  ),
  active = list(
    #' @field truth (`NULL`)\cr
    #'   An unsupervised task has no ground truth.
    truth = function(rhs) {
      assert_ro_binding(rhs)
      NULL
    },
    #' @field response (any)\cr
    #'   The predicted response.
    response = function(rhs) {
      assert_ro_binding(rhs)
      self$data$response
    },
    #' @field prob (any)\cr
    #'   The predicted probabilities.
    prob = function(rhs) {
      assert_ro_binding(rhs)
      self$data$prob
    }
  )
)

# `truth`, `response` and `prob` of a PredictionDataTorch can be anything the task's prediction
# encoder produced: a vector, a `data.table`, or an array of any dimensionality -- an autoencoder
# over images predicts an `(n, channels, height, width)` array, for instance.
# The only thing assumed about them is that their *first* dimension indexes the observations, so
# the prediction data methods below go through these helpers instead of indexing directly.
pt_nobs = function(x) {
  NROW(x)
}

pt_subset = function(x, i) {
  if (is.data.frame(x)) {
    return(x[i, , drop = FALSE])
  }
  d = dim(x)
  if (is.null(d)) {
    return(x[i])
  }
  # `x[i, , drop = FALSE]` with as many empty arguments as `x` has remaining dimensions
  index = rep(list(bquote()), length(d))
  index[[1L]] = i
  do.call("[", c(list(x), index, list(drop = FALSE)))
}

# binds arrays of more than two dimensions along their first dimension, which is what `rbind()`
# does for matrices and what `c()` does for vectors
pt_bind_arrays = function(xs) {
  d = dim(xs[[1L]])
  ns = map_int(xs, function(x) dim(x)[1L])
  walk(xs, function(x) {
    if (!identical(dim(x)[-1L], d[-1L])) {
      stopf("Cannot combine arrays of dimensions (%s) and (%s), they differ beyond the first dimension.", paste(d, collapse = ", "), paste(dim(x), collapse = ", ")) # nolint
    }
  })
  dim_out = c(sum(ns), d[-1L])
  out = array(vector(typeof(xs[[1L]]), prod(dim_out)), dim = dim_out,
    dimnames = c(list(NULL), dimnames(xs[[1L]])[-1L]))
  offset = 0L
  for (x in xs) {
    index = rep(list(bquote()), length(d))
    index[[1L]] = offset + seq_len(dim(x)[1L])
    out = do.call("[<-", c(list(out), index, list(value = x)))
    offset = offset + dim(x)[1L]
  }
  out
}

pt_combine = function(xs) {
  xs = xs[!map_lgl(xs, is.null)]
  # An element without observations contributes no rows, and its storage type carries no
  # information: an empty prediction (see `create_empty_prediction_data()`) may well be a bare
  # vector where the others are matrices. Drop them, so that the representative below is an
  # element that actually describes the storage.
  nonempty = map_int(xs, pt_nobs) > 0L
  if (any(nonempty)) {
    xs = xs[nonempty]
  }
  x = xs[[1L]]
  if (is.matrix(x)) {
    do.call(rbind, xs)
  } else if (is.array(x)) {
    pt_bind_arrays(xs)
  } else if (is.data.frame(x)) {
    rbindlist(xs, use.names = TRUE)
  } else if (is.factor(x)) {
    # unlist() would drop the levels and return the integer codes
    factor(unlist(lapply(xs, as.character), use.names = FALSE), levels = levels(x))
  } else {
    unlist(xs, use.names = FALSE)
  }
}

# turns one element of the prediction data into columns of `as.data.table(prediction)`
pt_as_columns = function(x, prefix) {
  # an observation of an array with more than two dimensions becomes one flat row of columns
  if (is.array(x) && length(dim(x)) > 2L) {
    x = matrix(x, nrow = NROW(x))
  }
  tab = if (is.matrix(x)) {
    as.data.table(x)
  } else if (is.data.frame(x)) {
    # copy(), because setnames() below renames by reference and `x` belongs to the prediction
    copy(as.data.table(x))
  } else {
    setnames(data.table(x), prefix)
  }
  if (ncol(tab) > 1L || !identical(names(tab), prefix)) {
    setnames(tab, paste0(prefix, ".", names(tab)))
  }
  tab
}

# The prediction data methods below are the same for the two generic torch task types, apart from
# the class they dispatch on and produce and the fact that an unsupervised prediction never carries
# a `truth`. They therefore share these implementations.
pt_elements = function(pdata) {
  intersect(c("truth", "response", "prob"), names(pdata))
}

pt_check = function(pdata) {
  n = length(assert_row_ids(pdata$row_ids))
  # deliberately lax: we only ensure that everything describes the same observations
  for (nm in pt_elements(pdata)) {
    n_nm = pt_nobs(pdata[[nm]])
    if (n_nm != n) {
      stopf("Element '%s' of the prediction data has %i observations, but %i row ids are given.", nm, n_nm, n) # nolint
    }
  }
  pdata
}

pt_is_missing = function(pdata) {
  response = pdata$response
  if (is.null(response)) {
    return(pdata$row_ids[0L])
  }
  miss = if (is.null(dim(response))) {
    is.na(response)
  } else {
    apply(response, 1L, anyNA)
  }
  pdata$row_ids[miss]
}

pt_empty_pdata = function(task, learner, cls) {
  pdata = list(row_ids = integer())

  truth = task$truth(integer(0))
  if (!is.null(truth)) {
    pdata$truth = truth
  }

  # What `response` and `prob` look like is decided by the task's prediction encoder, and an empty
  # prediction has to have the same storage type as a non-empty one so that the two can be
  # combined. Rather than guessing, we ask the encoder to encode an empty batch. A task that
  # cannot do this (no `output_dim`, or an encoder that rejects empty input) falls back to the
  # row ids and the truth alone.
  empty = try({
    encoded = encode_prediction(task, torch_zeros(0L, task$output_dim), learner$predict_type)
    encoded[intersect(mlr_reflections$learner_predict_types[[task$task_type]][[learner$predict_type]],
      names(encoded))]
  }, silent = TRUE)
  if (!inherits(empty, "try-error")) {
    pdata = c(pdata, discard(empty, is.null))
  }

  class(pdata) = c(cls, "PredictionData")
  pdata
}

pt_combine_pdata = function(dots, keep_duplicates, cls) {
  assert_list(dots, cls)
  assert_flag(keep_duplicates)
  if (length(dots) == 1L) {
    return(dots[[1L]])
  }

  elements = pt_elements(dots[[1L]])
  # Taking the elements from the first input alone would silently drop a `prob` that only the
  # later ones carry, so all inputs have to describe the same things.
  mismatch = map_lgl(dots, function(pdata) !setequal(pt_elements(pdata), elements))
  if (any(mismatch)) {
    stopf("Cannot combine prediction data with different predict types: %s vs %s.", str_collapse(elements), str_collapse(pt_elements(dots[[which(mismatch)[1L]]]))) # nolint
  }

  pdata = c(
    list(row_ids = do.call(c, map(dots, "row_ids"))),
    set_names(lapply(elements, function(nm) pt_combine(map(dots, nm))), elements)
  )

  if (!keep_duplicates) {
    keep = !duplicated(pdata$row_ids, fromLast = TRUE)
    pdata$row_ids = pdata$row_ids[keep]
    for (nm in elements) pdata[[nm]] = pt_subset(pdata[[nm]], keep)
  }

  class(pdata) = c(cls, "PredictionData")
  pdata
}

pt_filter = function(pdata, row_ids) {
  keep = pdata$row_ids %in% row_ids
  pdata$row_ids = pdata$row_ids[keep]
  for (nm in pt_elements(pdata)) {
    pdata[[nm]] = pt_subset(pdata[[nm]], keep)
  }
  pdata
}

pt_as_data_table = function(x) {
  tabs = c(
    list(data.table(row_ids = x$data$row_ids)),
    lapply(pt_elements(x$data), function(nm) pt_as_columns(x$data[[nm]], nm))
  )
  do.call(cbind, tabs)
}

#' @export
check_prediction_data.PredictionDataTorch = function(pdata, ...) { # nolint
  pt_check(pdata)
}

#' @export
check_prediction_data.PredictionDataTorchUnsupervised = function(pdata, ...) { # nolint
  if (!is.null(pdata$truth)) {
    stopf("Prediction data of an unsupervised torch task must not carry a `truth`.")
  }
  pt_check(pdata)
}

#' @export
is_missing_prediction_data.PredictionDataTorch = function(pdata, ...) { # nolint
  pt_is_missing(pdata)
}

#' @export
is_missing_prediction_data.PredictionDataTorchUnsupervised = function(pdata, ...) { # nolint
  pt_is_missing(pdata)
}

#' @export
as_prediction.PredictionDataTorch = function(x, check = TRUE, ...) { # nolint
  invoke(PredictionTorch$new, check = check, .args = x)
}

#' @export
as_prediction.PredictionDataTorchUnsupervised = function(x, check = TRUE, ...) { # nolint
  invoke(PredictionTorchUnsupervised$new, check = check, .args = x)
}

#' @export
create_empty_prediction_data.TaskTorch = function(task, learner) { # nolint
  pt_empty_pdata(task, learner, "PredictionDataTorch")
}

#' @export
create_empty_prediction_data.TaskTorchUnsupervised = function(task, learner) { # nolint
  pt_empty_pdata(task, learner, "PredictionDataTorchUnsupervised")
}

#' @export
c.PredictionDataTorch = function(..., keep_duplicates = TRUE) { # nolint
  pt_combine_pdata(list(...), keep_duplicates, "PredictionDataTorch")
}

#' @export
c.PredictionDataTorchUnsupervised = function(..., keep_duplicates = TRUE) { # nolint
  pt_combine_pdata(list(...), keep_duplicates, "PredictionDataTorchUnsupervised")
}

#' @export
filter_prediction_data.PredictionDataTorch = function(pdata, row_ids, ...) { # nolint
  pt_filter(pdata, row_ids)
}

#' @export
filter_prediction_data.PredictionDataTorchUnsupervised = function(pdata, row_ids, ...) { # nolint
  pt_filter(pdata, row_ids)
}

#' @export
as.data.table.PredictionTorch = function(x, ...) { # nolint
  pt_as_data_table(x)
}

#' @export
as.data.table.PredictionTorchUnsupervised = function(x, ...) { # nolint
  pt_as_data_table(x)
}
