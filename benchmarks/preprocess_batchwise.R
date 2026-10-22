# The torchvision transformations behind the deterministic preprocessing operators were applied to each element
# of a batch (rowwise = TRUE). This compares that with applying them to the whole batch, which gives the same
# result for the input dimensions where mlr3torch now does it (see batchwise() in R/preprocess.R).
# Batch of 32 images of size 3 x 64 x 64, CPU with one thread.
library(torch)
library(torchvision)
torch_set_num_threads(1)
torch_manual_seed(1)

per_element = function(fn, x, args) {
  torch_stack(lapply(seq_len(x$shape[1]), function(i) do.call(fn, c(list(x[i, ..]), args))), dim = 1L)
}
# median over 10 rounds of the mean time in ms of `n` calls of `f()`
ms = function(f, n = 10, rounds = 10) {
  f()
  median(replicate(rounds, {
    t0 = Sys.time()
    for (i in 1:n) f()
    1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / n
  }))
}

cases = list(
  trafo_adjust_gamma = list(transform_adjust_gamma, list(gamma = 0.5)),
  trafo_grayscale = list(transform_grayscale, list(num_output_channels = 1L)),
  trafo_rgb_to_grayscale = list(transform_rgb_to_grayscale, list()),
  trafo_normalize = list(transform_normalize, list(mean = c(0.5, 0.4, 0.3), std = c(0.2, 0.3, 0.4))),
  trafo_pad = list(transform_pad, list(padding = 3L, padding_mode = "reflect")),
  augment_hflip = list(transform_hflip, list()),
  augment_vflip = list(transform_vflip, list()),
  augment_crop = list(transform_crop, list(top = 3L, left = 5L, height = 20L, width = 30L)),
  augment_center_crop = list(transform_center_crop, list(size = 32L)),
  augment_resized_crop = list(transform_resized_crop, list(top = 3L, left = 5L, height = 30L, width = 30L, size = 16L))
)
x = torch_rand(32, 3, 64, 64)
for (id in names(cases)) {
  fn = cases[[id]][[1L]]
  args = cases[[id]][[2L]]
  expected = per_element(fn, x, args)
  same = torch_allclose(do.call(fn, c(list(x), args)), expected, atol = 1e-6)
  cat(sprintf("%-24s same: %-5s per element %6.2f ms   whole batch %5.2f ms\n", id, same,
    ms(function() per_element(fn, x, args)), ms(function() do.call(fn, c(list(x), args)))))
}

# Results, median over 10 rounds of 10 calls, one pinned CPU core:
# trafo_adjust_gamma       same: TRUE  per element  35.85 ms   whole batch  5.59 ms
# trafo_grayscale          same: TRUE  per element  73.24 ms   whole batch  4.87 ms
# trafo_rgb_to_grayscale   same: TRUE  per element  63.23 ms   whole batch  3.36 ms
# trafo_normalize          same: TRUE  per element  60.90 ms   whole batch  3.16 ms
# trafo_pad                same: TRUE  per element  37.72 ms   whole batch  1.96 ms
# augment_hflip            same: TRUE  per element   9.66 ms   whole batch  0.48 ms
# augment_vflip            same: TRUE  per element   9.62 ms   whole batch  0.60 ms
# augment_crop             same: TRUE  per element  15.32 ms   whole batch  0.33 ms
# augment_center_crop      same: TRUE  per element  37.16 ms   whole batch  0.93 ms
# augment_resized_crop     same: TRUE  per element  55.54 ms   whole batch  1.64 ms
# augment_rotate fails on the whole batch (in grid_sampler()) and stays per element.
# For trafo_normalize, mlr3torch uses its own batch-wise version (normalize_batch()), which also handles e.g. videos.
