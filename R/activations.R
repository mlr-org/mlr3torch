reglu = function(x) {
  assert_true(tail(x$shape, 1) %% 2 == 0)
  chunked = x$chunk(2, dim=-1)
  a = chunked[[1]]
  b = chunked[[2]]
  return(a * nnf_relu(b))
}

geglu = function(x) {
  assert_true(tail(x$shape, 1) %% 2 == 0)
  chunked = x$chunk(2, dim=-1)
  a = chunked[[1]]
  b = chunked[[2]]
  return(a * nnf_gelu(b))
}
