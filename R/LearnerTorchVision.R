#' @title Image Classification Networks from torchvision
#'
#' @name mlr_learners.torchvision
#'
#' @description
#' Well-known image classification networks from the
#' [`torchvision`](https://github.com/mlverse/torchvision) package.
#'
#' Each of these learners wraps one of the `model_*()` generators of `torchvision`.
#' The section *Available Learners* below lists for every learner the original architecture that it
#' implements, the paper that introduced this architecture, and the file in `torchvision` that
#' contains the implementation, so the architecture can be inspected directly.
#'
#' @eval torchvision_learner_section()
#'
#' @section Input Size:
#' The *Input Size* column of the table above states which spatial input sizes a network accepts.
#' Inputs do not have to be square, and any size at or above the stated minimum works.
#'
#' * `any`: the convolutional networks that pool their feature map to a fixed size before the
#'   classifier accept arbitrarily small inputs.
#' * A minimum (`at least NxN`): `AlexNet`, `VGG`, `ConvNeXt` and `Inception v3` shrink the feature
#'   map below one pixel for smaller inputs. Note that these are the sizes at which the networks
#'   still run, not the sizes at which they were trained, which is 224x224 for `AlexNet`, `VGG` and
#'   `ConvNeXt` and 299x299 for `Inception v3`.
#' * A fixed size (`NxN`): the vision transformers do not interpolate their position embeddings and
#'   `MaxViT` needs an input that its window partitioning divides evenly, so these require exactly
#'   the size that they were configured with.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorchImage`] and
#'
#' * `pretrained` :: `logical(1)`\cr
#'     Whether to use the pretrained model.
#'     The final linear layer will be replaced with a new `nn_linear` with the
#'     number of classes inferred from the [`Task`][mlr3::Task].
#'
#' `classif.inception_v3` additionally has
#'
#' * `aux_logits` :: `logical(1)`\cr
#'     Whether to enable the auxiliary classifier, which acts as a regularizer during training and
#'     is not applied when predicting. Defaults to `FALSE`.
#'     The configured loss is applied to the predictions of both classifiers and does not have to
#'     be changed for this.
#'     Note that the auxiliary classifier raises the minimum input size from 75x75 to 299x299,
#'     because it pools and convolves the feature map further than the main classifier does.
#' * `aux_weight` :: `numeric(1)`\cr
#'     The weight with which the loss of the auxiliary classifier is added to the loss of the main
#'     classifier. Defaults to `0.4`, the value used by the Inception v3 paper.
#'     Only has an effect if `aux_logits` is `TRUE`.
#'
#' @section Properties:
#' * Supported task types: `"classif"`
#' * Predict Types: `"response"` and `"prob"`
#' * Feature Types: `"lazy_tensor"`
#' * Required packages: `"mlr3torch"`, `"torch"`, `"torchvision"`
#' @template params_learner
#' @param name (`character(1)`)\cr
#'   The name of the network.
#' @param module_generator (`function(pretrained, num_classes)`)\cr
#'   Function that generates the network.
#' @param label (`character(1)`)\cr
#'   The label of the network.
#' @param jittable (`logical(1)`)\cr
#'   Whether to use jitting.
#' @eval torchvision_references()
#' @include LearnerTorchImage.R
#' @export
LearnerTorchVision = R6Class("LearnerTorchVision",
  inherit = LearnerTorchImage,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(name, module_generator, label, optimizer = NULL, loss = NULL,
      callbacks = list(), jittable = FALSE, extra_param_set = NULL,
      network_args = character(0)) { # nolint
      task_type = "classif"
      private$.module_generator = module_generator
      param_set = ps(
        pretrained = p_lgl(tags = c("required", "train"))
      )
      if (!is.null(extra_param_set)) {
        param_set = ps_union(list(param_set, extra_param_set))
        private$.network_args = assert_subset(network_args, param_set$ids())
      }
      param_set$values = list(pretrained = TRUE)
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".", name),
        param_set = param_set,
        man = paste0("mlr3torch::mlr_learners.torchvision"),
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        label = label,
        packages = "torchvision",
        jittable = jittable
      )
    }
  ),
  private = list(
    .module_generator = NULL,
    # ids of the parameters that are forwarded to the module generator
    .network_args = character(0),
    .network = function(task, param_vals) {
      nout = output_dim_for(task)
      args = param_vals[intersect(names(param_vals), private$.network_args)]
      if (param_vals$pretrained) {
        network = invoke(private$.module_generator, pretrained = TRUE, .args = args)
        return(replace_head(network, nout))
      }
      invoke(private$.module_generator, pretrained = FALSE, num_classes = nout, .args = args)
    },
    # With an enabled auxiliary classifier the network returns one prediction per classifier, so
    # the loss that the user configured is wrapped instead of being applied directly. `aux_logits`
    # only exists for the networks that have an auxiliary classifier, so this is a no-op otherwise.
    .loss_fn = function(task, param_vals) {
      loss_fn = super$.loss_fn(task, param_vals)
      if (isTRUE(param_vals$aux_logits)) {
        loss_fn = nn_aux_loss(loss_fn, aux_weight = param_vals$aux_weight %??% 0.4)
      }
      loss_fn
    },
    .additional_phash_input = function() {
      list(private$.module_generator, private$.network_args)
    }
  )
)

#' @export
replace_head.AlexNet = function(network, d_out) {
  network$classifier$`6` = torch::nn_linear(
    in_features = network$classifier$`6`$in_features,
    out_features = d_out,
    bias = TRUE
  )
  network
}

#' @export
replace_head.ConvNeXt = function(network, d_out) {
  network$head = nn_linear(network$head$in_features, d_out)
  network
}

#' @export
replace_head.efficientnet = function(network, d_out) {
  network$classifier$`1` = nn_linear(network$classifier$`1`$in_features, d_out)
  network
}

#' @export
replace_head.efficientnet_v2 = function(network, d_out) {
  network$classifier$`1` = nn_linear(network$classifier$`1`$in_features, d_out)
  network
}

#' @export
replace_head.Inception3 = function(network, d_out) {
  network$fc = nn_linear(network$fc$in_features, d_out)
  # the auxiliary classifier has a head of its own, which would otherwise keep predicting the
  # number of classes of the pretrained weights
  if (inherits(network$AuxLogits, "InceptionAux")) {
    network$AuxLogits$fc = nn_linear(network$AuxLogits$fc$in_features, d_out)
  }
  network
}

#' @export
replace_head.maxvit = function(network, d_out) {
  network$classifier$`5` = nn_linear(network$classifier$`5`$in_features, d_out)
  network
}

#' @export
replace_head.mobilenet_v2 = function(network, d_out) {
  network$classifier$`1` = nn_linear(1280, d_out)
  network
}

#' @export
replace_head.MobileNetV3 = function(network, d_out) {
  network$classifier$`3` = nn_linear(network$classifier$`3`$in_features, d_out)
  network
}

#' @export
replace_head.resnet = function(network, d_out) {
  in_features = network$fc$in_features
  network$fc = nn_linear(in_features, d_out)
  network
}

#' @export
replace_head.VGG = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vit_model = function(network, d_out) {
  network$head = nn_linear(network$head$in_features, d_out)
  network
}

# Wraps a loss so that it can be applied to a network that returns more than one prediction. The
# first prediction is the primary one, the remaining ones come from auxiliary classifiers and are
# added with weight `aux_weight`.
# This is only applied during training, where the auxiliary classifiers are active: the loss is
# not evaluated when predicting, and the validation scores are calculated from the predictions of
# the network, not from the loss.
nn_aux_loss = nn_module("nn_aux_loss",
  initialize = function(base_loss, aux_weight = 0.4) {
    self$base_loss = assert_class(base_loss, "nn_module")
    self$aux_weight = assert_number(aux_weight, lower = 0)
  },
  forward = function(input, target) {
    # indexing a tensor with [[ would silently select its first row instead of failing
    assert_list(input, min.len = 1L)
    loss = self$base_loss(input[[1L]], target)
    for (i in seq_along(input)[-1L]) {
      loss = loss + self$aux_weight * self$base_loss(input[[i]], target)
    }
    loss
  }
)

# In training mode, Inception v3 returns a list of (logits, aux_logits) when the auxiliary
# classifier is enabled, which the learner handles by wrapping the loss, see `.loss_fn()`. The
# auxiliary classifier is disabled by default, because it requires 299x299 inputs.
# When `pretrained` is TRUE, torchvision needs the auxiliary classifier to load the state dict,
# hence we can only remove it afterwards. Note that assigning `NULL` does not deregister a
# submodule, so it is replaced by `nn_identity()`, which drops its parameters. Both `aux_logits`
# and `AuxLogits` have to be reset, because `forward()` dispatches on the former and `.forward()`
# on the latter.
inception_v3_generator = function(pretrained = FALSE, aux_logits = FALSE, ...) {
  if (pretrained) {
    network = torchvision::model_inception_v3(pretrained = TRUE, ...)
  } else {
    network = torchvision::model_inception_v3(pretrained = FALSE, aux_logits = aux_logits,
      init_weights = FALSE, ...)
  }
  if (!aux_logits) {
    network$aux_logits = FALSE
    network$AuxLogits = nn_identity()
  }
  network
}

# The module returned by torchvision has no class of its own, so we add one to be able to dispatch
# replace_head() on it.
maxvit_generator = function(...) {
  network = torchvision::model_maxvit(...)
  class(network) = c("maxvit", class(network))
  network
}

# Single source of truth for the learner registrations below and for the documentation table
# generated by torchvision_learner_section().
#
# * id        : suffix of the learner id, e.g. "resnet18" for "classif.resnet18".
# * generator : name of the function that builds the network. Names starting with "model_" refer to
#               generators of torchvision, all others to generators of mlr3torch.
# * label     : label of the learner.
# * arch      : the original architecture that the network implements.
# * bib       : keys into `bibentries` of the paper(s) that introduced the architecture, separated
#               by ",".
# * file      : file in https://github.com/mlverse/torchvision that implements the network.
# * min_size  : smallest spatial input size that the network accepts, NA if it accepts any size.
# * exact_size: the only spatial input size that the network accepts, NA if it accepts more than
#               one size. At most one of `min_size` and `exact_size` is given.
# * jittable  : whether the network can be traced with `torch::jit_trace()`.
torchvision_models = local({
  tvm = function(id, generator, label, arch, bib, file, min_size = NA_integer_,
    exact_size = NA_integer_, jittable = TRUE) {
    list(id = id, generator = generator, label = label, arch = arch, bib = bib, file = file,
      min_size = min_size, exact_size = exact_size, jittable = jittable)
  }
  rbindlist(list(
    # torchvision implements the AlexNet variant of the "One weird trick" paper.
    tvm("alexnet", "model_alexnet", "AlexNet",
      "AlexNet", "krizhevsky2017imagenet,krizhevsky2014one", "models-alexnet.R",
      min_size = 63L),

    tvm("convnext_tiny_1k", "model_convnext_tiny_1k", "ConvNeXt-T (ImageNet-1k)",
      "ConvNeXt", "liu2022convnet", "models-convnext.R",
      min_size = 32L),
    tvm("convnext_tiny_22k", "model_convnext_tiny_22k", "ConvNeXt-T (ImageNet-22k)",
      "ConvNeXt", "liu2022convnet", "models-convnext.R",
      min_size = 32L),
    tvm("convnext_small_22k", "model_convnext_small_22k", "ConvNeXt-S (ImageNet-22k)",
      "ConvNeXt", "liu2022convnet", "models-convnext.R",
      min_size = 32L),
    # `classif.convnext_small_22k1k` is not registered, because `model_convnext_small_22k1k()`
    # cannot load its own pretrained weights: the model is fine-tuned on ImageNet-1k and its
    # weights have a 1000-class head, but torchvision builds it with `num_classes = 21841`, so
    # loading the state dict fails with "The size of tensor a (21841) must match the size of
    # tensor b (1000)". Passing `num_classes` does not help, because torchvision loads the state
    # dict with `strict = FALSE`, which `load_state_dict()` of torch for R does not support.
    # Re-enable this once torchvision has fixed the default.
    # tvm("convnext_small_22k1k", "model_convnext_small_22k1k",
    #   "ConvNeXt-S (ImageNet-22k, fine-tuned on ImageNet-1k)",
    #   "ConvNeXt", "liu2022convnet", "models-convnext.R",
    #   min_size = 32L),
    tvm("convnext_base_1k", "model_convnext_base_1k", "ConvNeXt-B (ImageNet-1k)",
      "ConvNeXt", "liu2022convnet", "models-convnext.R",
      min_size = 32L),
    tvm("convnext_base_22k", "model_convnext_base_22k", "ConvNeXt-B (ImageNet-22k)",
      "ConvNeXt", "liu2022convnet", "models-convnext.R",
      min_size = 32L),
    tvm("convnext_large_1k", "model_convnext_large_1k", "ConvNeXt-L (ImageNet-1k)",
      "ConvNeXt", "liu2022convnet", "models-convnext.R",
      min_size = 32L),
    tvm("convnext_large_22k", "model_convnext_large_22k", "ConvNeXt-L (ImageNet-22k)",
      "ConvNeXt", "liu2022convnet", "models-convnext.R",
      min_size = 32L),

    tvm("efficientnet_b0", "model_efficientnet_b0", "EfficientNet-B0",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_b1", "model_efficientnet_b1", "EfficientNet-B1",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_b2", "model_efficientnet_b2", "EfficientNet-B2",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_b3", "model_efficientnet_b3", "EfficientNet-B3",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_b4", "model_efficientnet_b4", "EfficientNet-B4",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_b5", "model_efficientnet_b5", "EfficientNet-B5",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_b6", "model_efficientnet_b6", "EfficientNet-B6",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_b7", "model_efficientnet_b7", "EfficientNet-B7",
      "EfficientNet", "tan2019efficientnet", "models-efficientnet.R"),
    tvm("efficientnet_v2_s", "model_efficientnet_v2_s", "EfficientNetV2-S",
      "EfficientNetV2", "tan2021efficientnetv2", "models-efficientnetv2.R"),
    tvm("efficientnet_v2_m", "model_efficientnet_v2_m", "EfficientNetV2-M",
      "EfficientNetV2", "tan2021efficientnetv2", "models-efficientnetv2.R"),
    tvm("efficientnet_v2_l", "model_efficientnet_v2_l", "EfficientNetV2-L",
      "EfficientNetV2", "tan2021efficientnetv2", "models-efficientnetv2.R"),

    # With an enabled auxiliary classifier the network returns a list of predictions, which
    # torch::jit_trace() cannot represent, so Inception v3 is not traceable.
    tvm("inception_v3", "inception_v3_generator", "Inception v3",
      "Inception v3", "szegedy2016rethinking", "models-inception.R", min_size = 75L,
      jittable = FALSE),

    tvm("maxvit", "maxvit_generator", "MaxViT-T",
      "MaxViT", "tu2022maxvit", "models-maxvit.R", exact_size = 224L),

    tvm("mobilenet_v2", "model_mobilenet_v2", "MobileNetV2",
      "MobileNetV2", "sandler2018mobilenetv2", "models-mobilenetv2.R"),
    tvm("mobilenet_v3_large", "model_mobilenet_v3_large", "MobileNetV3-Large",
      "MobileNetV3", "howard2019searching", "models-mobilenetv3.R"),
    tvm("mobilenet_v3_small", "model_mobilenet_v3_small", "MobileNetV3-Small",
      "MobileNetV3", "howard2019searching", "models-mobilenetv3.R"),

    tvm("resnet18", "model_resnet18", "ResNet-18",
      "ResNet", "he2016deep", "models-resnet.R"),
    tvm("resnet34", "model_resnet34", "ResNet-34",
      "ResNet", "he2016deep", "models-resnet.R"),
    tvm("resnet50", "model_resnet50", "ResNet-50",
      "ResNet", "he2016deep", "models-resnet.R"),
    tvm("resnet101", "model_resnet101", "ResNet-101",
      "ResNet", "he2016deep", "models-resnet.R"),
    tvm("resnet152", "model_resnet152", "ResNet-152",
      "ResNet", "he2016deep", "models-resnet.R"),
    tvm("resnext50_32x4d", "model_resnext50_32x4d", "ResNeXt-50 32x4d",
      "ResNeXt", "he2016deep,xie2017aggregated", "models-resnet.R"),
    tvm("resnext101_32x8d", "model_resnext101_32x8d", "ResNeXt-101 32x8d",
      "ResNeXt", "he2016deep,xie2017aggregated", "models-resnet.R"),
    tvm("wide_resnet50_2", "model_wide_resnet50_2", "Wide ResNet-50-2",
      "Wide ResNet", "he2016deep,zagoruyko2016wide", "models-resnet.R"),
    tvm("wide_resnet101_2", "model_wide_resnet101_2", "Wide ResNet-101-2",
      "Wide ResNet", "he2016deep,zagoruyko2016wide", "models-resnet.R"),

    tvm("vgg11", "model_vgg11", "VGG-11",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),
    tvm("vgg11_bn", "model_vgg11_bn", "VGG-11 with batch normalization",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),
    tvm("vgg13", "model_vgg13", "VGG-13",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),
    tvm("vgg13_bn", "model_vgg13_bn", "VGG-13 with batch normalization",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),
    tvm("vgg16", "model_vgg16", "VGG-16",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),
    tvm("vgg16_bn", "model_vgg16_bn", "VGG-16 with batch normalization",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),
    tvm("vgg19", "model_vgg19", "VGG-19",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),
    tvm("vgg19_bn", "model_vgg19_bn", "VGG-19 with batch normalization",
      "VGG", "simonyan2014very", "models-vgg.R",
      min_size = 32L),

    # torch::jit_trace() cannot trace the attention blocks of the vision transformers.
    tvm("vit_b_16", "model_vit_b_16", "ViT-B/16",
      "Vision Transformer (ViT)", "dosovitskiy2021image", "models-vit.R",
      exact_size = 224L, jittable = FALSE),
    tvm("vit_b_32", "model_vit_b_32", "ViT-B/32",
      "Vision Transformer (ViT)", "dosovitskiy2021image", "models-vit.R",
      exact_size = 224L, jittable = FALSE),
    tvm("vit_l_16", "model_vit_l_16", "ViT-L/16",
      "Vision Transformer (ViT)", "dosovitskiy2021image", "models-vit.R",
      exact_size = 224L, jittable = FALSE),
    tvm("vit_l_32", "model_vit_l_32", "ViT-L/32",
      "Vision Transformer (ViT)", "dosovitskiy2021image", "models-vit.R",
      exact_size = 224L, jittable = FALSE),
    tvm("vit_h_14", "model_vit_h_14", "ViT-H/14",
      "Vision Transformer (ViT)", "dosovitskiy2021image", "models-vit.R",
      exact_size = 518L, jittable = FALSE)
  ))
})

# Generators are resolved lazily, because torchvision is only a suggested package.
torchvision_module_generator = function(generator) {
  if (startsWith(generator, "model_")) {
    getExportedValue("torchvision", generator)
  } else {
    get(generator, envir = asNamespace("mlr3torch"))
  }
}

torchvision_source_url = function(file) {
  paste0("https://github.com/mlverse/torchvision/blob/main/R/", file)
}

torchvision_bib_keys = function(bib) {
  strsplit(bib, ",", fixed = TRUE)[[1L]]
}

# Parameters that some of the networks have in addition to `pretrained`. `network_args` lists
# those that are forwarded to the module generator, the remaining ones are interpreted by the
# learner itself, see the `.network()` and `.loss_fn()` methods of `LearnerTorchVision`.
torchvision_extra_params = list(
  inception_v3 = function() {
    list(
      param_set = ps(
        aux_logits = p_lgl(default = FALSE, tags = "train"),
        aux_weight = p_dbl(lower = 0, default = 0.4, tags = "train")
      ),
      network_args = "aux_logits"
    )
  }
)

#' @include aaa.R
local({
  for (i in seq_len(nrow(torchvision_models))) {
    # the inner local() gives every learner its own environment, so that the constructor below
    # refers to the values of the current row and not to those of the last one
    local({
      id = torchvision_models$id[i]
      generator = torchvision_models$generator[i]
      label = torchvision_models$label[i]
      jittable = torchvision_models$jittable[i]
      extra_params = torchvision_extra_params[[id]]
      register_learner(paste0("classif.", id),
        function(loss = NULL, optimizer = NULL, callbacks = list()) {
          extra = if (!is.null(extra_params)) extra_params()
          LearnerTorchVision$new(id, torchvision_module_generator(generator), label,
            loss = loss, optimizer = optimizer, callbacks = callbacks, jittable = jittable,
            extra_param_set = extra$param_set, network_args = extra$network_args %??% character(0))
        }
      )
    })
  }
})

# Human-readable form of the `min_size` / `exact_size` columns of `torchvision_models`.
torchvision_input_size = function(min_size, exact_size) {
  if (!is.na(exact_size)) {
    sprintf("%1$ix%1$i", exact_size)
  } else if (!is.na(min_size)) {
    sprintf("at least %1$ix%1$i", min_size)
  } else {
    "any"
  }
}

# Roxygen block listing which original model each learner implements, which paper introduced it,
# and where its implementation can be found. Used via `@eval`.
torchvision_learner_section = function() {
  rows = map_chr(seq_len(nrow(torchvision_models)), function(i) {
    keys = torchvision_bib_keys(torchvision_models$bib[i])
    sprintf("| `classif.%s` | %s | %s | [%s](%s) | %s |",
      torchvision_models$id[i],
      torchvision_models$arch[i],
      invoke(cite_bib, .args = as.list(keys), bibentries = bibentries),
      torchvision_models$file[i],
      torchvision_source_url(torchvision_models$file[i]),
      torchvision_input_size(torchvision_models$min_size[i], torchvision_models$exact_size[i])
    )
  })
  c(
    "@section Available Learners:",
    "The table below lists for each learner the original architecture that it implements, the",
    "paper that introduced this architecture, and the file in `torchvision` that contains the",
    "implementation.",
    "",
    "| Learner | Architecture | Reference | Implementation | Input Size |",
    "| --- | --- | --- | --- | --- |",
    rows
  )
}

# All papers that introduced one of the architectures above. Used via `@eval`.
torchvision_references = function() {
  keys = unique(unlist(lapply(torchvision_models$bib, torchvision_bib_keys)))
  c("@references", map_chr(keys, function(key) format_bib(key, bibentries = bibentries)))
}
