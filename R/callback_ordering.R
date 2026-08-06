# Why the built-in callbacks sit at the weight they do, keyed by that weight. Only the prose lives
# here: the weights themselves, and which callback has which, are read from the callbacks when the
# documentation is built, so the table cannot claim a weight the code does not use.
callback_weight_reasons = list(
  "-200" = "changes which parameters of the network are trained, which the batch that is about to run and everything that inspects or saves the network must already see", # nolint
  "0" = "custom callbacks, unless they ask for something else",
  "100" = "decides on `ctx$last_scores_valid`, which a custom callback can still change in the same stage, and sets `ctx$terminate` before the callbacks that report on the epoch run", # nolint
  "200" = "records `ctx$last_scores_train` and `ctx$last_scores_valid` of the epoch that just ran",
  "300" = "logs the same scores to disk",
  "400" = "its summary closes the epoch, so it is printed after what the other callbacks have to say", # nolint
  "500" = "stepping the schedule changes the learning rate for the *next* epoch or batch, so it happens after everything that reports on the one that just ran, and before the checkpoint saves the optimizer", # nolint
  "Inf" = "saves the network, the optimizer and the other callbacks' `$state_dict()`s, so everything that still changes them must have run" # nolint
)

# The rows of the ordering table: every callback that declares a weight, grouped by it.
# `mlr3torch_callbacks` covers the ones with a dictionary entry; the other two are the default of
# CallbackSet itself and early stopping, which the learner builds from its `patience` parameter.
callback_weight_table = function() {
  entries = rbind(
    as.data.table(mlr3torch_callbacks)[, list(weight, name = sprintf("`%s`", get("key")))],
    data.table(
      weight = CallbackSetEarlyStopping$public_fields$weight,
      name = "early stopping (the learner's `patience`)"
    ),
    data.table(weight = CallbackSet$public_fields$weight, name = "*default*")
  )
  entries = entries[, list(name = paste(sort(get("name")), collapse = ", ")), by = "weight"]
  setorderv(entries, "weight")[]
}

# The 'Ordering' section of CallbackSet, see @eval there.
callback_ordering_section = function() {
  entries = callback_weight_table()
  rows = sprintf("| `%s` | %s | %s |",
    # trim, as format() would pad the numbers to a common width inside the code spans
    format(entries$weight, trim = TRUE),
    entries$name,
    map_chr(as.character(entries$weight), function(w) callback_weight_reasons[[w]] %??% "")
  )
  c(
    "@section Ordering:",
    "Within a stage, callbacks are called in the order in which they were passed to the learner.",
    "A callback can override this via its `$weight` field: callbacks with a higher weight are",
    "called after those with a lower one, and callbacks with the same weight keep the order in",
    "which they were passed.",
    "",
    "This matters for callbacks that build on what the others did.",
    "The built-in callbacks therefore declare the weights below, spaced so that a custom callback",
    "can be slotted between any two of them.",
    "The table is generated from the callbacks themselves, so it cannot go stale; the same numbers",
    "are available programmatically as the `weight` column of",
    "`as.data.table(`[`mlr3torch_callbacks`]`)`, and printing a [`TorchCallback`] shows its weight.",
    "",
    "| weight | callback | why |",
    "| ---: | --- | --- |",
    rows,
    "",
    "Two callbacks that need to run in a fixed order relative to each other should have different",
    "weights, as equal ones only keep the order they happen to be passed in.",
    "",
    "[`CallbackSetCheckpoint`] is the one exception to the rule that equal weights keep the order",
    "the callbacks were passed in: it is always called last within its stage, also when another",
    "callback has weight `Inf` as well.",
    "Every weight, `Inf` included, is otherwise free to use."
  )
}
