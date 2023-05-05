# Just a simple benchmark that compares different approaches to merging tensors.

library(torch)
n1 = 1000
n2 = 1
n3 = 3000

compare = function(..., n) {

  # generate the input tensors
  tensors = lapply(seq(n), function(i) torch_randn(..., device = "cpu"))

  merge_stack = function(...) {
    torch_sum(torch_stack(torch_broadcast_tensors(list(...)), dim = 1L), dim = 1L)
  }

  merge_reduce = function(...) {
    Reduce(torch_add, list(...))
  }

  bench::mark(
    stack = do.call(merge_stack, args = tensors),
    reduce = do.call(merge_reduce, args = tensors)
  )
}

compare(100, 1, 1, 300, n = 5)
compare(30, 50, 30, n = 5)

# the reduce approach is faster
