# #' @export
# as.data.table.DictionaryTask = function(x, ..., objects = FALSE) { # nolint
#   assert_flag(objects)
#   # FIXME: This should be in mlr3
#   # Also check whether the object is downloaded and if yes we can actually construct the task.
#
#   setkeyv(map_dtr(x$keys(), function(key) {
#     t = tryCatch(x$get(key),
#       missingDefaultError = function(e) NULL)
#     if (is.null(t) || key %in% names(mlr3torch_image_tasks)) {
#       return(list(key = key))
#     }
#
#     feats = translate_types(t$feature_types$type)
#     insert_named(
#       c(list(key = key, label = t$label, task_type = t$task_type, nrow = t$nrow, ncol = t$ncol, properties = list(t$properties)), table(feats)),
#       if (objects) list(object = list(t))
#     )
#   }, .fill = TRUE), "key")[]
# }
