#
nn = function(.key, ...) {
  invoke(po, .obj = paste0("nn_", .key), .args = insert_named(list(id = .key), list(...)))
}
