# Defining an Architecture

In this vignette, we will show how to build neural network architectures
as `mlr3pipelines::Graphs`s. We will create a simple CNN for the
tiny-imagenet task, which is a subset of well-known Imagenet benchmark.

``` r
library(mlr3torch)
imagenet = tsk("tiny_imagenet")
imagenet
#> 
#> ── <TaskClassif> (110000x2): ImageNet Subset ───────────────────────────────────
#> • Target: class
#> • Target classes: abacus (0%), academic gown, academic robe, judge's robe (0%),
#> acorn (0%), African elephant, Loxodonta africana (0%), albatross, mollymawk
#> (0%), alp (0%), altar (0%), American alligator, Alligator mississipiensis (0%),
#> American lobster, Northern lobster, Maine lobster, Homarus americanus (0%),
#> apron (0%) + 190 more
#> • Properties: multiclass
#> • Features (1):
#>   • lt (1): image
```

The central ingredients for creating such graphs are `PipeOpTorch`
operators.

To mark the entry-point of the neural network, we use a
`PipeOpTorchIngress`, for which three different flavors exist:

- `po("torch_ingress_num")` for numeric data
- `po("torch_ingress_categ")` for categorical columns
- `po("torch_ingress_ltnsr")` for `lazy_tensor`s

Because the imagenet task contains only one feature of type
`lazy_tensor`, we go for the last option:

``` r
architecture = po("torch_ingress_ltnsr")
```

We now define a relatively simple convolutional neural network. Note
that in the code below `po("nn_relu_1")` is equivalent to
`po("nn_relu", id = "nn_linear_1")`. This is needed, because
[`mlr3pipelines::Graph`](https://mlr3pipelines.mlr-org.com/reference/Graph.html)s
require that each `PipeOp` has a unique ID.

What we can further notice is that we don’t have to specify the input
dimension for the convolutional layers, which are inferred from the task
during `$train()`ing. This means that our `Learner` can be applied to
tasks with different image sizes, each time building up the correct
network structure.

``` r
architecture = architecture %>>%
  po("nn_conv2d_1", out_channels = 64, kernel_size = 11, stride = 4, padding = 2) %>>%
  po("nn_relu_1", inplace = TRUE) %>>%
  po("nn_max_pool2d_1", kernel_size = 3, stride = 2) %>>%
  po("nn_conv2d_2", out_channels = 192, kernel_size = 5, padding = 2) %>>%
  po("nn_relu_2", inplace = TRUE) %>>%
  po("nn_max_pool2d_2", kernel_size = 3, stride = 2)
```

We can now continue with specifying the classification part of the
network, which is a dense network that repeats a layer twice:

``` r
dense_layer = po("nn_dropout") %>>%
  po("nn_linear", out_features = 4096) %>>%
  po("nn_relu_6")
```

In order to repeat a segment from a network multiple times, we can use
`po("nn_block")`, which we here repeat twice. Then, we follow with the
output head of the network, where we don’t have to specify the number of
classes, as they can also be inferred from the task

``` r
classifier = po("nn_block", dense_layer, n_blocks = 2L) %>>%
  po("nn_head")
```

Next, we can combine the convolutional part with the dense head:

``` r
architecture = architecture %>>%
  po("nn_flatten") %>>%
  classifier
```

Below, we display the network:

``` r
architecture$plot(html = TRUE)
```

To turn this network architecture into an
[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) what
is left to do is to configure the loss, optimizer, callbacks, and
training arguments, which we do now: We use the standard cross-entropy
loss, SGD as the optimizer and checkpoint our model every 20 epochs.

``` r
checkpoint = tempfile()
architecture = architecture %>>%
  po("torch_loss", t_loss("cross_entropy")) %>>%
  po("torch_optimizer", t_opt("sgd", lr=0.01)) %>>%
  po("torch_callbacks", 
    t_clbk("checkpoint", freq = 20, path = checkpoint)) %>>%
  po("torch_model_classif",
    batch_size = 32, epochs = 100L, device = "cuda")

cnn = as_learner(architecture)
cnn$id = "cnn"
```

This created `Learner` now exposes all configuration options of the
individual `PipeOp`s in its `$param_set`, from which we show only a
subset for readability:

``` r
as.data.table(cnn$param_set)[c(32, 34, 42), 1:4]
#>                       id    class lower upper
#>                   <char>   <char> <num> <num>
#> 1:     nn_block.n_blocks ParamInt     0   Inf
#> 2: nn_block.nn_dropout.p ParamDbl     0     1
#> 3:  torch_loss.reduction ParamFct    NA    NA
```

We can still change them, or if we wanted to, even tune them! Below, we
increase the number of blocks and latent dimension of the dense part, as
well as change the learning rate of the SGD optimizer.

``` r
cnn$param_set$set_values(
  nn_block.n_blocks = 4L,
  nn_block.nn_linear.out_features = 4096 * 2,
  torch_optimizer.lr = 0.2
)
```

Finally, we train the learner on the task:

``` r
cnn$train(imagenet)
```
