# Code that was used to generate the toytask

library(mlr3torch)
task = tsk("tiny_imagenet")
dat = task$data()
idx = seq(from = 1, by = 500, length.out = 200)
dat = dat[idx, ]
print(length(unique(dat$class)))
uris = dat$image

for (uri in uris) {
  command = sprintf("cp %s ~/mlr/mlr3torch/inst/toytask/", uri)
  system(
    command
  )
}

labels = dat$class
