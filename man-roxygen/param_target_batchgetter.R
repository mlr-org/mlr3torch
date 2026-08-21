#' @param target_batchgetter (`function()` or `NULL`)\cr
#'   Converts the target columns of a batch into the target tensor `y` that the loss is applied to.
#'   Takes an argument `data`, a [`data.table`][data.table::data.table] with only the target columns,
#'   and optionally an argument `x`, the named list of feature tensors of the batch, which is what a
#'   target that is a function of the input needs, see [`get_target_batchgetter()`].
#'   <%= if (exists("target_batchgetter_null")) target_batchgetter_null else "If `NULL` (default), it is taken from the task via [`get_target_batchgetter()`], which the built-in task types provide, but a [`TaskTorch`] only if it has no target at all, in which case the batches have no `y` element." %>
