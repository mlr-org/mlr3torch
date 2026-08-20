#' @title Prediction Object for a Generic Torch Task
#'
#' @description
#' The [`Prediction`][mlr3::Prediction] object returned by learners that were trained on a
#' [`TaskTorch`].
#'
#' Because a [`TaskTorch`] can represent very different learning problems, this class does not
#' prescribe much about how `truth`, `response`, `prob` and `se` are stored.
#' Each of them may be an atomic vector, a `matrix()`, a [`data.table`][data.table::data.table], an
#' array of any dimensionality, or a [`lazy_tensor`], i.e., anything whose *first* dimension indexes the
#' observations.
#'
#' @template params_prediction_torch
#' @param truth (any)\cr
#'   The ground truth, i.e. what `task$truth()` returned.
#' @param response (any)\cr
#'   The predicted response.
#' @param prob (any)\cr
#'   The predicted probabilities.
#' @param se (any)\cr
#'   The standard errors of the prediction.
#' @param lazy_tensor ([`lazy_tensor`])\cr
#'   The output of the network, see the predict type `"lazy_tensor"` of [`LearnerTorch`].
#' @param weights (`numeric()` or `NULL`)\cr
#'   The measure weights of the predicted observations, i.e. the `weights_measure` column of the
#'   task. `mlr3` fills this in, so it rarely has to be passed by hand.
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
      lazy_tensor = NULL, weights = NULL, check = TRUE) {
      pdata = discard(list(row_ids = row_ids, truth = truth, response = response, prob = prob,
        se = se, lazy_tensor = lazy_tensor, weights = weights), is.null)
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
    },
    #' @field lazy_tensor ([`lazy_tensor`])\cr
    #'   The output of the network, for the predict type `"lazy_tensor"`.
    lazy_tensor = function(rhs) {
      assert_ro_binding(rhs)
      self$data$lazy_tensor
    }
  )
)

pt_predict_types = c("response", "prob", "se", "lazy_tensor")
pt_elements = c("truth", pt_predict_types, "weights")

# One cell per observation, each holding that observation's own array. The class is only a hook for
# printing: `as.data.table()` on a prediction exists mostly to print it, and a `data.table` pastes
# the contents of a list column -- megabytes of numbers for a batch of images -- unless the column's
# class has a `format_col()` method. `lazy_tensor` hooks into the same generic.
pt_arrays = function(x) {
  cells = lapply(seq_len(NROW(x)), function(i) array(pt_subset(x, i), dim = dim(x)[-1L]))
  structure(cells, class = c("pt_arrays", "list"))
}

#' @export
format.pt_arrays = function(x, ...) { # nolint
  map_chr(x, function(el) sprintf("<array[%s]>", paste0(dim(el), collapse = "x")))
}

#' @exportS3Method data.table::format_col
format_col.pt_arrays = function(x, ...) { # nolint
  format(x, ...)
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

pt_combine = function(xs) {
  xs = xs[!map_lgl(xs, is.null)]
  nonempty = map_int(xs, NROW) > 0L
  if (any(nonempty)) {
    xs = xs[nonempty]
  }
  x = xs[[1L]]
  if (is.matrix(x)) {
    do.call(rbind, xs)
  } else if (is.array(x)) {
    # `rbind()` only understands two dimensions -- it would flatten the rest into columns
    rbind_arrays(xs)
  } else if (is.data.frame(x)) {
    rbindlist(xs, use.names = TRUE)
  } else if (inherits(x, "lazy_tensor")) {
    # FIXME: general concatenation of lazy tensors is not allowed (only when they have teh same DataDescriptor),
    # so we have a special case here
    if (!length(x)) {
      lazy_tensor()
    } else if (length(unique(map_chr(xs, function(xi) dd(xi)$hash))) == 1L) {
      do.call(c, xs)
    } else {
      walk(xs, function(xi) {
        if (!inherits(dd(xi)$dataset, "in_memory_tensor_dataset")) {
          stopf("Cannot combine lazy tensors that were built from different datasets unless each already holds its tensors in memory, because combining them materialises them -- and materialising a lazy tensor that reads its data on demand would read all of it. Combine the parts yourself if that is what you want.") # nolint
        }
      })
      as_lazy_tensor(torch_cat(lapply(xs, materialize, rbind = TRUE), dim = 1L))
    }
  } else if (is.factor(x)) {
    rbindlist(lapply(xs, function(xi) data.table(x = xi)))$x
  } else {
    unname(do.call(c, xs))
  }
}

#' @export
check_prediction_data.PredictionDataTorch = function(pdata, ...) { # nolint
  n = length(assert_row_ids(pdata$row_ids))
  for (nm in intersect(pt_elements, names(pdata))) {
    el = pdata[[nm]]
    n_nm = NROW(el)
    if (n_nm != n) {
      stopf("Element '%s' of the prediction data has %i observations, but %i row ids are given.", nm, n_nm, n) # nolint
    }
    # a `lazy_tensor` and a `data.table` are lists too, and both are fine: it is the bare one that
    # is refused, see the class description
    if (is.list(el) && !is.data.frame(el) && !inherits(el, "lazy_tensor")) {
      stopf("Element '%s' of the prediction data is a bare `list()`. Store a prediction that is one value per observation in an atomic vector, a matrix or an array, and one that is a tensor per observation in a `lazy_tensor`.", nm) # nolint
    }
  }
  pdata
}

#' @export
is_missing_prediction_data.PredictionDataTorch = function(pdata, ...) { # nolint
  response = pdata$response
  # Only a response with one value per observation can say that an observation was not predicted.
  # Anything wider -- a matrix, an array, a `data.table`, a `lazy_tensor` -- would first have to
  # decide what a partially missing observation is, and a `lazy_tensor` could not answer at all
  # without materialising the whole prediction, so none of them report missing predictions.
  if (is.null(response) || !is.atomic(response) || !is.null(dim(response))) {
    return(pdata$row_ids[0L])
  }
  pdata$row_ids[is.na(response)]
}

#' @export
as_prediction.PredictionDataTorch = function(x, check = TRUE, ...) { # nolint
  invoke(PredictionTorch$new, check = check, .args = x)
}

#' @export
as_prediction_data.prediction_torch = function(x, task, row_ids = task$row_ids, check = TRUE, ...) { # nolint
  class(x) = "list"
  pdata = as_prediction_data(x, task = task, row_ids = row_ids, check = FALSE, ...)
  if (is.null(pdata$truth)) {
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

  if (learner$predict_type == "lazy_tensor") {
    # `as_lazy_tensor()` cannot build one from a tensor without rows, and there is nothing to ask
    # the encoder about: this predict type never consults it
    pdata$lazy_tensor = lazy_tensor()
  } else {
    empty = get_private(learner)$.encode_prediction(
      network_output = torch_zeros(0L, output_dim_for(task)), task = task)
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
      el = x$data[[nm]]
      # `prob` becomes one column per class, the way `mlr3` tables a classification prediction.
      # Every other array becomes one cell per observation, holding that observation's own array:
      # its width is a property of the prediction rather than of the problem, and the
      # (n, 3, 224, 224) reconstruction of an autoencoder over images would be 150528 columns, a
      # table that cannot even be printed.
      if (is.array(el) && nm != "prob") {
        el = pt_arrays(el)
      }
      tab = if (is.matrix(el)) {
        as.data.table(el)
      } else if (is.data.frame(el)) {
        # copy(), because setnames() below renames by reference and `el` belongs to the prediction
        copy(as.data.table(el))
      } else {
        setnames(data.table(el), nm)
      }
      if (ncol(tab) > 1L || !identical(names(tab), nm)) {
        setnames(tab, paste0(nm, ".", names(tab)))
      }
      tab
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
