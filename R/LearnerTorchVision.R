#' @title AlexNet Image Classifier
#'
#' @name mlr_learners.torchvision
#'
#' @description
#' Classic image classification networks from `torchvision`.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorchImage`] and
#'
#' * `pretrained` :: `logical(1)`\cr
#'     Whether to use the pretrained model.
#'     The final linear layer will be replaced with a new `nn_linear` with the
#'     number of classes inferred from the [`Task`][mlr3::Task].
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
#'#' @references
#' `r format_bib("krizhevsky2017imagenet")`
#' `r format_bib("sandler2018mobilenetv2")`
#' `r format_bib("he2016deep")`
#' `r format_bib("simonyan2014very")`
#' @include LearnerTorchImage.R
#' @export
LearnerTorchVision = R6Class("LearnerTorchVision",
  inherit = LearnerTorchImage,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(name, module_generator, label, optimizer = NULL, loss = NULL, callbacks = list()) { # nolint
      task_type = "classif"
      private$.module_generator = module_generator
      param_set = ps(
        pretrained = p_lgl(tags = c("required", "train"))
      )
      param_set$values = list(pretrained = TRUE)
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".", name),
        param_set = param_set,
        man = paste0("mlr3torch::mlr_learners.torchvision"),
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        label = label
      )
    }
  ),
  private = list(
    .module_generator = NULL,
    .network = function(task, param_vals) {
      nout = get_nout(task)
      if (param_vals$pretrained) {
        network = replace_head(private$.module_generator(pretrained = TRUE), nout)
        return(network)
      }
      private$.module_generator(pretrained = FALSE, num_classes = nout)
    },
    .additional_phash_input = function() {
      list(private$.module_generator)
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

# #' @export
# replace_head.Inception3 = function(network, d_out) {
#   network$fc = nn_linear(2048, d_out)
#   network
# }

#' @export
replace_head.mobilenet_v2 = function(network, d_out) {
  network$classifier$`1` = nn_linear(1280, d_out)
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

#' @include zzz.R
register_learner("classif.alexnet", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("alexnet", torchvision::model_alexnet, "AlexNet",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

# register_learner("classif.inception_v3", 
#   function(loss = NULL, optimizer = NULL, callbacks = list()) {
#     LearnerTorchVision$new("inception_v3", torchvision::model_inception_v3, "Inception V3",
#     loss = loss, optimizer = optimizer, callbacks = callbacks)
#   }
# )

register_learner("classif.mobilenet_v2", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("mobilenet_v2", torchvision::model_mobilenet_v2, "Mobilenet V2",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.resnet18", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("resnet18", torchvision::model_resnet18, "ResNet-18",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.resnet34", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("resnet34", torchvision::model_resnet34, "ResNet-34",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.resnet50", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("resnet50", torchvision::model_resnet50, "ResNet-50",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.resnet101", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("resnet101", torchvision::model_resnet101, "ResNet-101",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.resnet152", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("resnet152", torchvision::model_resnet152, "ResNet-152",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.resnext101_32x8d", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("resnext101_32x8d", torchvision::model_resnext101_32x8d, "ResNeXt-101 32x8d",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.resnext50_32x4d", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("resnext50_32x4d", torchvision::model_resnext50_32x4d, "ResNeXt-50 32x4d",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg11", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg11", torchvision::model_vgg11, "VGG 11",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg11_bn", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg11_bn", torchvision::model_vgg11_bn, "VGG 11",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg13", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg13", torchvision::model_vgg13, "VGG 13",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg13_bn", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg13_bn", torchvision::model_vgg13_bn, "VGG 13",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg16", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg16", torchvision::model_vgg16, "VGG 16",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg16_bn", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg16_bn", torchvision::model_vgg16_bn, "VGG 16",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg19", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg19", torchvision::model_vgg19, "VGG 19",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)

register_learner("classif.vgg19_bn", 
  function(loss = NULL, optimizer = NULL, callbacks = list()) {
    LearnerTorchVision$new("vgg19_bn", torchvision::model_vgg19_bn, "VGG 19",
      loss = loss, optimizer = optimizer, callbacks = callbacks)
  }
)
