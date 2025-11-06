# AlexNet Image Classifier

Classic image classification networks from `torchvision`.

## Parameters

Parameters from
[`LearnerTorchImage`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_image.md)
and

- `pretrained` :: `logical(1)`  
  Whether to use the pretrained model. The final linear layer will be
  replaced with a new `nn_linear` with the number of classes inferred
  from the [`Task`](https://mlr3.mlr-org.com/reference/Task.html).

## Properties

- Supported task types: `"classif"`

- Predict Types: `"response"` and `"prob"`

- Feature Types: `"lazy_tensor"`

- Required packages: `"mlr3torch"`, `"torch"`, `"torchvision"`

## References

Krizhevsky, Alex, Sutskever, Ilya, Hinton, E. G (2017). “Imagenet
classification with deep convolutional neural networks.” *Communications
of the ACM*, **60**(6), 84–90. Sandler, Mark, Howard, Andrew, Zhu,
Menglong, Zhmoginov, Andrey, Chen, Liang-Chieh (2018). “Mobilenetv2:
Inverted residuals and linear bottlenecks.” In *Proceedings of the IEEE
conference on computer vision and pattern recognition*, 4510–4520. He,
Kaiming, Zhang, Xiangyu, Ren, Shaoqing, Sun, Jian (2016). “Deep residual
learning for image recognition.” In *Proceedings of the IEEE conference
on computer vision and pattern recognition*, 770–778. Simonyan, Karen,
Zisserman, Andrew (2014). “Very deep convolutional networks for
large-scale image recognition.” *arXiv preprint arXiv:1409.1556*.

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`mlr3torch::LearnerTorch`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch.md)
-\>
[`mlr3torch::LearnerTorchImage`](https://mlr3torch.mlr-org.com/reference/mlr_learners_torch_image.md)
-\> `LearnerTorchVision`

## Methods

### Public methods

- [`LearnerTorchVision$new()`](#method-LearnerTorchVision-new)

- [`LearnerTorchVision$clone()`](#method-LearnerTorchVision-clone)

Inherited methods

- [`mlr3::Learner$base_learner()`](https://mlr3.mlr-org.com/reference/Learner.html#method-base_learner)
- [`mlr3::Learner$configure()`](https://mlr3.mlr-org.com/reference/Learner.html#method-configure)
- [`mlr3::Learner$encapsulate()`](https://mlr3.mlr-org.com/reference/Learner.html#method-encapsulate)
- [`mlr3::Learner$help()`](https://mlr3.mlr-org.com/reference/Learner.html#method-help)
- [`mlr3::Learner$predict()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict)
- [`mlr3::Learner$predict_newdata()`](https://mlr3.mlr-org.com/reference/Learner.html#method-predict_newdata)
- [`mlr3::Learner$reset()`](https://mlr3.mlr-org.com/reference/Learner.html#method-reset)
- [`mlr3::Learner$selected_features()`](https://mlr3.mlr-org.com/reference/Learner.html#method-selected_features)
- [`mlr3::Learner$train()`](https://mlr3.mlr-org.com/reference/Learner.html#method-train)
- [`mlr3torch::LearnerTorch$dataset()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-dataset)
- [`mlr3torch::LearnerTorch$format()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-format)
- [`mlr3torch::LearnerTorch$marshal()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-marshal)
- [`mlr3torch::LearnerTorch$print()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-print)
- [`mlr3torch::LearnerTorch$unmarshal()`](https://mlr3torch.mlr-org.com/reference/LearnerTorch.html#method-unmarshal)

------------------------------------------------------------------------

### Method `new()`

Creates a new instance of this
[R6](https://r6.r-lib.org/reference/R6Class.html) class.

#### Usage

    LearnerTorchVision$new(
      name,
      module_generator,
      label,
      optimizer = NULL,
      loss = NULL,
      callbacks = list(),
      jittable = FALSE
    )

#### Arguments

- `name`:

  (`character(1)`)  
  The name of the network.

- `module_generator`:

  (`function(pretrained, num_classes)`)  
  Function that generates the network.

- `label`:

  (`character(1)`)  
  The label of the network.

- `optimizer`:

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/reference/TorchOptimizer.md))  
  The optimizer to use for training. Per default, *adam* is used.

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/reference/TorchLoss.md))  
  The loss used to train the network. Per default, *mse* is used for
  regression and *cross_entropy* for classification.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/reference/TorchCallback.md)s)  
  The callbacks. Must have unique ids.

- `jittable`:

  (`logical(1)`)  
  Whether to use jitting.

------------------------------------------------------------------------

### Method `clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchVision$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
