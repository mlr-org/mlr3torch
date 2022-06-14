reset_parameters = function(nn) {
  f = function(x) {
    if (!is.null(x$reset_parameters)) {
      x$reset_parameters()
    }
  }
  nn$apply(f)
}

reset_running_stats = function(nn) {
  f = function(x) {
    if (!is.null(x$reset_running_stats)) {
      x$reset_running_stats()
    }
  }
  nn$apply(f)
}
