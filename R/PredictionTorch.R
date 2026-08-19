#' @title Prediction Object for a Generic Torch Task
#'
#' @description
#' The [`Prediction`][mlr3::Prediction] object returned by learners that were trained on a
#' [`TaskTorch`].
#'
#' Because a [`TaskTorch`] can represent very different learning problems, this class does not
#' prescribe how `truth`, `response`, `prob` and `se` are stored.
#' Each of them may be an atomic vector, a `matrix()`, a [`data.table`][data.table::data.table], an
#' array of any dimensionality, or a `list()` with one element per observation -- whatever the
#' task's `prediction_encoder` produced.
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
#' @param se (any)\cr
#'   The standard errors of the prediction.
#' @param weights (`numeric()` or `NULL`)\cr
#'   The measure weights of the predicted observations, i.e. the `weights_measure` column of the
#'   task. `mlr3` fills this in, so it rarely has to be passed by hand.
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
      truth = if (!is.null(task)) task$truth(row_ids), response = NULL, prob = NULL, se = NULL,
      weights = NULL, check = TRUE) {
      pdata = discard(list(row_ids = row_ids, truth = truth, response = response, prob = prob,
        se = se, weights = weights), is.null)
      class(pdata) = c("PredictionDataTorch", "PredictionData")
      if (check) {
        pdata = check_prediction_data(pdata)
      }

      self$task_type = "torch"
      self$man = "mlr3torch::PredictionTorch"
      self$data = pdata
      self$predict_types = intersect(pt_predict_types, names(pdata))
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
    },
    #' @field se (any)\cr
    #'   The standard errors of the prediction.
    se = function(rhs) {
      assert_ro_binding(rhs)
      self$data$se
    }
  )
)

# The elements a prediction of a generic torch task can carry. `truth` comes from the task, the
# others are whatever the prediction encoder returned, filtered by the learner's predict type.
pt_predict_types = c("response", "prob", "se")
# `weights` is not a predict type but is subset and combined like one; `mlr3` puts it there for a
# task with a `weights_measure` column and `Measure$score()` reads it back out
pt_elements = c("truth", pt_predict_types, "weights")

# `truth`, `response` and `prob` of a PredictionDataTorch can be anything the task's prediction
# encoder produced: a vector, a `data.table`, an array of any dimensionality -- an autoencoder over
# images predicts an `(n, channels, height, width)` array, for instance -- or a list, which is what
# is left when the observations do not share a shape at all, as for a `lazy_tensor`.
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
  out = array(vector(typeof(xs[[1L]]), prod(dim_out)), dim = dim_out)
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
    # `unlist()` here would strip the class of anything that is not a bare atomic vector: a
    # `lazy_tensor` (a classed list) is flattened into its internals, a `Date` is demoted to the
    # numbers underneath it. `c()` dispatches, so it keeps whatever the prediction encoder built.
    unname(do.call(c, xs))
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

#' @export
check_prediction_data.PredictionDataTorch = function(pdata, ...) { # nolint
  n = length(assert_row_ids(pdata$row_ids))
  # deliberately lax: we only ensure that everything describes the same observations
  for (nm in intersect(pt_elements, names(pdata))) {
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
  miss = if (is.array(response) || is.data.frame(response)) {
    # `is.na()` on an array is an array of the same shape, so an observation is asked about along
    # its first margin instead -- for a matrix this is the usual row-wise question
    apply(response, 1L, anyNA)
  } else if (is.list(response)) {
    # a list stores one arbitrary object per observation, so only that object can be asked
    map_lgl(response, anyNA)
  } else {
    is.na(response)
  }
  pdata$row_ids[miss]
}

#' @export
as_prediction.PredictionDataTorch = function(x, check = TRUE, ...) { # nolint
  invoke(PredictionTorch$new, check = check, .args = x)
}

#' @title Prediction Data of a Torch Learner
#' @description
#' `mlr3` copies `task$truth()` into the prediction data only for a
#' [`TaskSupervised`][mlr3::TaskSupervised], and a [`TaskTorch`] is not one -- it may have any
#' number of target columns, including none.
#' The private `.predict()` method of [`LearnerTorch`] therefore gives what it returns this class,
#' so that this method runs instead of the one for a plain `list()` and adds the ground truth.
#' @param x (`prediction_torch`)\cr
#'   What `.predict()` returned, i.e. a named `list()` of prediction types.
#' @param task ([`Task`][mlr3::Task])\cr
#'   The task that was predicted on.
#' @param row_ids (`integer()`)\cr
#'   The predicted rows.
#' @param check (`logical(1)`)\cr
#'   Whether to check the assembled prediction data.
#' @param ... (any)\cr
#'   Passed on.
#' @return [`PredictionData`][mlr3::PredictionData]
# Everything that turns a network output into prediction data goes through here, so that the
# method below runs. Tagging at the point of use rather than in `.encode_prediction()`, which a
# learner may overwrite -- an overwritten one would silently produce a prediction without a truth.
as_prediction_data_torch = function(x, task, row_ids = task$row_ids, check = TRUE) {
  class(x) = c("prediction_torch", "list")
  as_prediction_data(x, task = task, row_ids = row_ids, check = check)
}

#' @export
as_prediction_data.prediction_torch = function(x, task, row_ids = task$row_ids, check = TRUE, ...) { # nolint
  class(x) = "list"
  # the truth is not among the predict types, so it cannot be passed in and has to be set afterwards
  pdata = as_prediction_data(x, task = task, row_ids = row_ids, check = FALSE, ...)
  if (is.null(pdata$truth)) {
    # `NULL` for a task without target columns, which removes the element rather than setting it
    pdata$truth = task$truth(row_ids)
  }
  if (check) {
    pdata = check_prediction_data(pdata)
  }
  pdata
}

#' @export
create_empty_prediction_data.TaskTorch = function(task, learner) { # nolint
  pdata = list(row_ids = integer())

  truth = task$truth(integer(0))
  if (!is.null(truth)) {
    pdata$truth = truth
  }
  if ("weights_measure" %chin% task$properties) {
    pdata$weights = numeric()
  }

  # An empty prediction has to have the same storage as a non-empty one so that the two can be
  # combined, and only the prediction encoder knows what that is, so we ask it to encode an empty
  # batch. This goes through the learner rather than calling `encode_prediction()` directly,
  # because a learner may encode predictions itself and then never consult the task.
  empty = try({
    encoded = get_private(learner)$.encode_prediction(
      network_output = torch_zeros(0L, output_dim_for(task)), task = task)
    encoded[intersect(mlr_reflections$learner_predict_types$torch[[learner$predict_type]],
      names(encoded))]
  }, silent = TRUE)
  if (inherits(empty, "try-error")) {
    # Degrading silently would leave the empty prediction without a `response`, and the only symptom
    # would be a `different predict types` error once it is combined with a real prediction.
    warningf("Could not build an empty prediction for task '%s', so it carries only row ids and the truth: %s", task$id, trimws(conditionMessage(attr(empty, "condition")))) # nolint
  } else {
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

  elements = intersect(pt_elements, names(dots[[1L]]))
  # Taking the elements from the first input alone would silently drop a `prob` that only the
  # later ones carry, so all inputs have to describe the same things.
  mismatch = map_lgl(dots, function(pdata) {
    !setequal(intersect(pt_elements, names(pdata)), elements)
  })
  if (any(mismatch)) {
    stopf("Cannot combine prediction data with different predict types: %s vs %s.", str_collapse(elements), str_collapse(intersect(pt_elements, names(dots[[which(mismatch)[1L]]])))) # nolint
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
  # `mlr3::c.Prediction()` ends in `as_prediction(pdata, check = FALSE)`, so an element that does
  # not combine cleanly would otherwise travel on as a prediction whose parts have different
  # lengths, and only surface much later as a nonsensical score.
  check_prediction_data(pdata)
}

#' @export
filter_prediction_data.PredictionDataTorch = function(pdata, row_ids, ...) { # nolint
  keep = pdata$row_ids %in% row_ids
  pdata$row_ids = pdata$row_ids[keep]
  for (nm in intersect(pt_elements, names(pdata))) {
    pdata[[nm]] = pt_subset(pdata[[nm]], keep)
  }
  pdata
}

#' @export
as.data.table.PredictionTorch = function(x, ...) { # nolint
  tabs = c(
    list(data.table(row_ids = x$data$row_ids)),
    lapply(intersect(c("truth", pt_predict_types), names(x$data)), function(nm) {
      pt_as_columns(x$data[[nm]], nm)
    })
  )
  # `cbind()` recycles a shorter table instead of complaining, which would turn a malformed
  # prediction into a table with duplicated observations rather than an error
  nrows = map_int(tabs, nrow)
  if (!all(nrows == nrows[1L])) {
    stopf("Prediction has %i row ids, but its elements have %s observations.", nrows[1L], str_collapse(nrows[-1L])) # nolint
  }
  do.call(cbind, tabs)
}
