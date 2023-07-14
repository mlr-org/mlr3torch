The data in `./tests/testthat/assets/nano_cats_vs_dogs` comes from the training set of `torchvision::dogs_vs_cats_dataset()`.
The names were left unchanged.

The data in `./tests/testthat/assets/nano_mnist` was downloaded from: https://github.com/teavanist/MNIST-JPG.
6 random pictures (2 from class 1, 2 and 3) were taken and the true label prepended to the name.

The data in `./tests/testthat/assets/nano_imagenet` was created as follows.
It contains 5 "wok" and 5 "torch" images

```{r}
task = tsk("tiny_imagenet")
  task = tsk("tiny_imagenet")

  dt = task$data()
  dt = dt[class %in% c("wok", "torch"), ]
  dt = dt[,.SD[1:5], by = class]

  for (i in seq_len(nrow(dt))) {
    path = as.character(dt[i, "image"][[1L]][1L])
    new_name = paste0(dt[i, "class"][[1L]][1L], "_", basename(path))
    system(sprintf("cp %s /home/sebi/mlr/mlr3torch/tests/testthat/assets/nano_imagenet/%s", path, new_name))
  }
```
