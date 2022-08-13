make_serialization = function(network, optimizer, loss_fn) {
  serialize_tensors <- function(x, f) {
      lapply(x, function(x) {
          if (inherits(x, "torch_tensor")) {
            torch:::tensor_to_raw_vector_with_class(x)
          }
          else if (is.list(x)) {
              serialize_tensors(x)
          }
          else {
              x
          }
      })
  }
  f = function(x) {
    serialized = serialize_tensors(x)
    list(values = serialized, type = "list", version = torch:::use_ser_version())

  }
  list(
    state_dict = list(
      network = f(network$state_dict()),
      optimizer = f(optimizer$state_dict()),
      loss_fn = f(loss_fn$state_dict())
    ),
    prototype
  )
}
  serialize_tensors <- function(x, f) {
      lapply(x, function(x) {
          if (inherits(x, "torch_tensor")) {
            torch:::tensor_to_raw_vector_with_class(x)
          }
          else if (is.list(x)) {
              serialize_tensors(x)
          }
          else {
              x
          }
      })
  }

torch_load_list = function(obj, device = NULL) {
  reload = function(values) {
    lapply(values, function(x) {
      if (inherits(x, "torch_serialized_tensor")) {
        torch:::load_tensor_from_raw(x, device = device)
      } else if (is.list(x)) {
        reload(x)
      }
      else {
        x
      }
    })
  }
  reload(obj$values)
}

load_broken_state_dict = function(module, state_dict) {

}
