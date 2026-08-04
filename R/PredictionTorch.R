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

      self$task_type = "torch"
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

# `truth`, `response` and `prob` of a PredictionDataTorch can be vectors, matrices or data.tables,
# so the prediction data methods below go through these three helpers instead of indexing directly.
pt_nobs = function(x) {
  if (is.matrix(x) || is.data.frame(x)) nrow(x) else length(x)
}

pt_subset = function(x, i) {
  if (is.matrix(x) || is.data.frame(x)) x[i, , drop = FALSE] else x[i]
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

#' @export
check_prediction_data.PredictionDataTorch = function(pdata, ...) { # nolint
  n = length(assert_row_ids(pdata$row_ids))
  # deliberately lax: we only ensure that everything describes the same observations
  for (nm in intersect(c("truth", "response", "prob"), names(pdata))) {
    n_nm = pt_nobs(pdata[[nm]])
    if (n_nm != n) {
      stopf("Element '%s' of the prediction data has %i observations, but %i row ids are given.", nm, n_nm, n) # nolint
    }
  }
  pdata
}

#' @export
is_missing_prediction_data.PredictionDataTorch = function(pdata, ...) { # nolint
  response = pdata$response
  if (is.null(response)) {
    return(pdata$row_ids[0L])
  }
  miss = if (is.matrix(response) || is.data.frame(response)) {
    apply(response, 1L, anyNA)
  } else {
    is.na(response)
  }
  pdata$row_ids[miss]
}

#' @export
as_prediction.PredictionDataTorch = function(x, check = TRUE, ...) { # nolint
  invoke(PredictionTorch$new, check = check, .args = x)
}

#' @export
create_empty_prediction_data.TaskTorch = function(task, learner) { # nolint
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
    encoded[intersect(mlr_reflections$learner_predict_types$torch[[learner$predict_type]],
      names(encoded))]
  }, silent = TRUE)
  if (!inherits(empty, "try-error")) {
    pdata = c(pdata, discard(empty, is.null))
  }

  class(pdata) = c("PredictionDataTorch", "PredictionData")
  pdata
}

#' @export
c.PredictionDataTorch = function(..., keep_duplicates = TRUE) { # nolint
  dots = list(...)
  assert_list(dots, "PredictionDataTorch")
  assert_flag(keep_duplicates)
  if (length(dots) == 1L) {
    return(dots[[1L]])
  }

  elements = intersect(c("truth", "response", "prob"), names(dots[[1L]]))
  # Taking the elements from the first input alone would silently drop a `prob` that only the
  # later ones carry, so all inputs have to describe the same things.
  mismatch = map_lgl(dots, function(pdata) {
    !setequal(intersect(c("truth", "response", "prob"), names(pdata)), elements)
  })
  if (any(mismatch)) {
    stopf("Cannot combine prediction data with different predict types: %s vs %s.", str_collapse(elements), str_collapse(intersect(c("truth", "response", "prob"), names(dots[[which(mismatch)[1L]]])))) # nolint
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

  class(pdata) = c("PredictionDataTorch", "PredictionData")
  pdata
}

#' @export
filter_prediction_data.PredictionDataTorch = function(pdata, row_ids, ...) { # nolint
  keep = pdata$row_ids %in% row_ids
  pdata$row_ids = pdata$row_ids[keep]
  for (nm in intersect(c("truth", "response", "prob"), names(pdata))) {
    pdata[[nm]] = pt_subset(pdata[[nm]], keep)
  }
  pdata
}

#' @export
as.data.table.PredictionTorch = function(x, ...) { # nolint
  tabs = c(
    list(data.table(row_ids = x$data$row_ids)),
    lapply(intersect(c("truth", "response", "prob"), names(x$data)), function(nm) {
      pt_as_columns(x$data[[nm]], nm)
    })
  )
  do.call(cbind, tabs)
}
