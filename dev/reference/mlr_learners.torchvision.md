# Image Classification Networks from torchvision

Well-known image classification networks from the
[`torchvision`](https://github.com/mlverse/torchvision) package.

Each of these learners wraps one of the `model_*()` generators of
`torchvision`. The section *Available Learners* below lists for every
learner the original architecture that it implements, the paper that
introduced this architecture, and the file in `torchvision` that
contains the implementation, so the architecture can be inspected
directly.

## Available Learners

The table below lists for each learner the original architecture that it
implements, the paper that introduced this architecture, and the file in
`torchvision` that contains the implementation.

|  |  |  |  |
|----|----|----|----|
| Learner | Architecture | Reference | Implementation |
| `classif.alexnet` | AlexNet | Krizhevsky (2017) and Krizhevsky (2014) | [models-alexnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-alexnet.R) |
| `classif.convnext_tiny_1k` | ConvNeXt | Liu (2022) | [models-convnext.R](https://github.com/mlverse/torchvision/blob/main/R/models-convnext.R) |
| `classif.convnext_tiny_22k` | ConvNeXt | Liu (2022) | [models-convnext.R](https://github.com/mlverse/torchvision/blob/main/R/models-convnext.R) |
| `classif.convnext_small_22k` | ConvNeXt | Liu (2022) | [models-convnext.R](https://github.com/mlverse/torchvision/blob/main/R/models-convnext.R) |
| `classif.convnext_base_1k` | ConvNeXt | Liu (2022) | [models-convnext.R](https://github.com/mlverse/torchvision/blob/main/R/models-convnext.R) |
| `classif.convnext_base_22k` | ConvNeXt | Liu (2022) | [models-convnext.R](https://github.com/mlverse/torchvision/blob/main/R/models-convnext.R) |
| `classif.convnext_large_1k` | ConvNeXt | Liu (2022) | [models-convnext.R](https://github.com/mlverse/torchvision/blob/main/R/models-convnext.R) |
| `classif.convnext_large_22k` | ConvNeXt | Liu (2022) | [models-convnext.R](https://github.com/mlverse/torchvision/blob/main/R/models-convnext.R) |
| `classif.efficientnet_b0` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_b1` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_b2` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_b3` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_b4` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_b5` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_b6` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_b7` | EfficientNet | Tan (2019) | [models-efficientnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnet.R) |
| `classif.efficientnet_v2_s` | EfficientNetV2 | Tan (2021) | [models-efficientnetv2.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnetv2.R) |
| `classif.efficientnet_v2_m` | EfficientNetV2 | Tan (2021) | [models-efficientnetv2.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnetv2.R) |
| `classif.efficientnet_v2_l` | EfficientNetV2 | Tan (2021) | [models-efficientnetv2.R](https://github.com/mlverse/torchvision/blob/main/R/models-efficientnetv2.R) |
| `classif.inception_v3` | Inception v3 | Szegedy (2016) | [models-inception.R](https://github.com/mlverse/torchvision/blob/main/R/models-inception.R) |
| `classif.maxvit` | MaxViT | Tu (2022) | [models-maxvit.R](https://github.com/mlverse/torchvision/blob/main/R/models-maxvit.R) |
| `classif.mobilenet_v2` | MobileNetV2 | Sandler (2018) | [models-mobilenetv2.R](https://github.com/mlverse/torchvision/blob/main/R/models-mobilenetv2.R) |
| `classif.mobilenet_v3_large` | MobileNetV3 | Howard (2019) | [models-mobilenetv3.R](https://github.com/mlverse/torchvision/blob/main/R/models-mobilenetv3.R) |
| `classif.mobilenet_v3_small` | MobileNetV3 | Howard (2019) | [models-mobilenetv3.R](https://github.com/mlverse/torchvision/blob/main/R/models-mobilenetv3.R) |
| `classif.resnet18` | ResNet | He (2016) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.resnet34` | ResNet | He (2016) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.resnet50` | ResNet | He (2016) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.resnet101` | ResNet | He (2016) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.resnet152` | ResNet | He (2016) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.resnext50_32x4d` | ResNeXt | He (2016) and Xie (2017) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.resnext101_32x8d` | ResNeXt | He (2016) and Xie (2017) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.wide_resnet50_2` | Wide ResNet | He (2016) and Zagoruyko (2016) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.wide_resnet101_2` | Wide ResNet | He (2016) and Zagoruyko (2016) | [models-resnet.R](https://github.com/mlverse/torchvision/blob/main/R/models-resnet.R) |
| `classif.vgg11` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vgg11_bn` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vgg13` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vgg13_bn` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vgg16` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vgg16_bn` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vgg19` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vgg19_bn` | VGG | Simonyan (2014) | [models-vgg.R](https://github.com/mlverse/torchvision/blob/main/R/models-vgg.R) |
| `classif.vit_b_16` | Vision Transformer (ViT) | Dosovitskiy (2021) | [models-vit.R](https://github.com/mlverse/torchvision/blob/main/R/models-vit.R) |
| `classif.vit_b_32` | Vision Transformer (ViT) | Dosovitskiy (2021) | [models-vit.R](https://github.com/mlverse/torchvision/blob/main/R/models-vit.R) |
| `classif.vit_l_16` | Vision Transformer (ViT) | Dosovitskiy (2021) | [models-vit.R](https://github.com/mlverse/torchvision/blob/main/R/models-vit.R) |
| `classif.vit_l_32` | Vision Transformer (ViT) | Dosovitskiy (2021) | [models-vit.R](https://github.com/mlverse/torchvision/blob/main/R/models-vit.R) |
| `classif.vit_h_14` | Vision Transformer (ViT) | Dosovitskiy (2021) | [models-vit.R](https://github.com/mlverse/torchvision/blob/main/R/models-vit.R) |

## Parameters

Parameters from
[`LearnerTorchImage`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md)
and

- `pretrained` :: `logical(1)`  
  Whether to use the pretrained model. The final linear layer will be
  replaced with a new `nn_linear` with the number of classes inferred
  from the [`Task`](https://mlr3.mlr-org.com/reference/Task.html).

`classif.inception_v3` additionally has

- `aux_logits` :: `logical(1)`  
  Whether to enable the auxiliary classifier, which acts as a
  regularizer during training and is not applied when predicting.
  Defaults to `FALSE`. The configured loss is applied to the predictions
  of both classifiers and does not have to be changed for this. Note
  that the auxiliary classifier raises the minimum input size from 75x75
  to 299x299, because it pools and convolves the feature map further
  than the main classifier does.

- `aux_weight` :: `numeric(1)`  
  The weight with which the loss of the auxiliary classifier is added to
  the loss of the main classifier. Defaults to `0.4`, the value used by
  the Inception v3 paper. Only has an effect if `aux_logits` is `TRUE`.

## Properties

- Supported task types: `"classif"`

- Predict Types: `"response"` and `"prob"`

- Feature Types: `"lazy_tensor"`

- Required packages: `"mlr3torch"`, `"torch"`, `"torchvision"`

## References

Krizhevsky A, Sutskever I, Hinton G (2017). “Imagenet classification
with deep convolutional neural networks.” *Communications of the ACM*,
**60**(6), 84–90. Krizhevsky A (2014). “One weird trick for
parallelizing convolutional neural networks.” *arXiv preprint
arXiv:1404.5997*. Liu Z, Mao H, Wu C, Feichtenhofer C, Darrell T, Xie S
(2022). “A ConvNet for the 2020s.” In *Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR)*,
11976–11986. Tan M, Le Q (2019). “EfficientNet: Rethinking model scaling
for convolutional neural networks.” In *Proceedings of the 36th
International Conference on Machine Learning (ICML)*, 6105–6114. PMLR.
Tan M, Le Q (2021). “EfficientNetV2: Smaller models and faster
training.” In *Proceedings of the 38th International Conference on
Machine Learning (ICML)*, 10096–10106. PMLR. Szegedy C, Vanhoucke V,
Ioffe S, Shlens J, Wojna Z (2016). “Rethinking the inception
architecture for computer vision.” In *Proceedings of the IEEE
conference on computer vision and pattern recognition*, 2818–2826. Tu Z,
Talebi H, Zhang H, Yang F, Milanfar P, Bovik A, Li Y (2022). “MaxViT:
Multi-axis vision transformer.” In *Proceedings of the European
Conference on Computer Vision (ECCV)*, 459–479. Sandler M, Howard A, Zhu
M, Zhmoginov A, Chen L (2018). “Mobilenetv2: Inverted residuals and
linear bottlenecks.” In *Proceedings of the IEEE conference on computer
vision and pattern recognition*, 4510–4520. Howard A, Sandler M, Chu G,
Chen L, Chen B, Tan M, Wang W, Zhu Y, Pang R, Vasudevan V, Le Q, Adam H
(2019). “Searching for MobileNetV3.” In *Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV)*, 1314–1324. He K,
Zhang X, Ren S, Sun J (2016). “Deep residual learning for image
recognition.” In *Proceedings of the IEEE conference on computer vision
and pattern recognition*, 770–778. Xie S, Girshick R, Dollár P, Tu Z, He
K (2017). “Aggregated residual transformations for deep neural
networks.” In *Proceedings of the IEEE conference on computer vision and
pattern recognition*, 1492–1500. Zagoruyko S, Komodakis N (2016). “Wide
residual networks.” In *Proceedings of the British Machine Vision
Conference (BMVC)*. 1605.07146, <https://arxiv.org/abs/1605.07146>.
Simonyan K, Zisserman A (2014). “Very deep convolutional networks for
large-scale image recognition.” *arXiv preprint arXiv:1409.1556*.
Dosovitskiy A, Beyer L, Kolesnikov A, Weissenborn D, Zhai X, Unterthiner
T, Dehghani M, Minderer M, Heigold G, Gelly S, Uszkoreit J, Houlsby N
(2021). “An image is worth 16x16 words: Transformers for image
recognition at scale.” In *The Ninth International Conference on
Learning Representations (ICLR)*. 2010.11929,
<https://arxiv.org/abs/2010.11929>.

## Super classes

[`mlr3::Learner`](https://mlr3.mlr-org.com/reference/Learner.html) -\>
[`LearnerTorch`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch.md)
-\>
[`LearnerTorchImage`](https://mlr3torch.mlr-org.com/dev/reference/mlr_learners_torch_image.md)
-\> `LearnerTorchVision`

## Methods

### Public methods

- [`LearnerTorchVision$new()`](#method-LearnerTorchVision-initialize)

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
- [`LearnerTorch$dataset()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-dataset)
- [`LearnerTorch$format()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-format)
- [`LearnerTorch$marshal()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-marshal)
- [`LearnerTorch$print()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-print)
- [`LearnerTorch$unmarshal()`](https://mlr3torch.mlr-org.com/dev/reference/LearnerTorch.html#method-unmarshal)

------------------------------------------------------------------------

### `LearnerTorchVision$new()`

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
      jittable = FALSE,
      extra_param_set = NULL,
      network_args = character(0)
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

  ([`TorchOptimizer`](https://mlr3torch.mlr-org.com/dev/reference/TorchOptimizer.md))  
  The optimizer to use for training. Per default, *adam* is used.

- `loss`:

  ([`TorchLoss`](https://mlr3torch.mlr-org.com/dev/reference/TorchLoss.md))  
  The loss used to train the network. Per default, *mse* is used for
  regression and *cross_entropy* for classification.

- `callbacks`:

  ([`list()`](https://rdrr.io/r/base/list.html) of
  [`TorchCallback`](https://mlr3torch.mlr-org.com/dev/reference/TorchCallback.md)s)  
  The callbacks. Must have unique ids.

- `jittable`:

  (`logical(1)`)  
  Whether to use jitting.

- `extra_param_set`:

  ([`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html) or
  `NULL`)  
  Parameters that this network has in addition to `pretrained`, or
  `NULL` if it has none. They are added to the learner's
  [`ParamSet`](https://paradox.mlr-org.com/reference/ParamSet.html).

- `network_args`:

  ([`character()`](https://rdrr.io/r/base/character.html))  
  The ids of those parameters of `extra_param_set` that are passed on to
  `module_generator`. The remaining ones are interpreted by the learner
  itself, such as `aux_weight`, which is used to wrap the loss and never
  reaches the network.

------------------------------------------------------------------------

### `LearnerTorchVision$clone()`

The objects of this class are cloneable with this method.

#### Usage

    LearnerTorchVision$clone(deep = FALSE)

#### Arguments

- `deep`:

  Whether to make a deep clone.
