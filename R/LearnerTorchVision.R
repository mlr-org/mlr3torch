#' @title AlexNet Image Classifier
#'
#' @name mlr_learners.torchvision
#'
#' @description
#' Neural networks from {torchvision}.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorchImage`] and
#'
#' * `pretrained` :: `logical(1)`\cr
#'   Whether to use the pretrained model.
#' @section Properties:
#' * Supported task types: `"classif"`
#' * Predict Types: `"response"` and `"prob"`
#' * Feature Types: `"lazy_tensor"`
#' * Required packages: `"mlr3torch"`, `"torch"`, `"torchvision"`
#'
#' @references `r format_bib(
#'   "krizhevsky2017imagenet"
#' )`
#'
#' @include LearnerTorchImage.R
#' @export
LearnerTorchVision = R6Class("LearnerTorchVision",
  inherit = LearnerTorchImage,
  public = list(
    initialize = function(name, module_generator, pretrained_classes, label,
      optimizer = NULL, loss = NULL, callbacks = list()) {

      task_type = "classif"
      private$.pretrained_classes = assert_int(pretrained_classes, lower = 2L)
      private$.module_generator = module_generator
      param_set = ps(
        pretrained = p_lgl(tags = c("required", "train"))
      )

      param_set$values = list(pretrained = TRUE)
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".", name),
        param_set = param_set,
        man = paste0("mlr3torch::mlr_learners.", name),
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        label = label
      )
       
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nout = get_nout(task)
      if (param_vals$pretrained) {
        network = replace_head(private$.module_generator(petrained = TRUE), nout)

        return(network)
      }
      assert_true(isTRUE(all.equal(nout, privat$.pretrained_classes)))

      torchvision::model_alexnet(pretrained = FALSE, num_classes = nout)
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
replace_head.Inception3 = function(network, d_out) {
  network$fc = nn_linear(2048, d_out)
  network
}

#' @export
replace_head.mobilenet_v2 = function(network, d_out) {
  network$classifier$`1` = torch::nn_linear(1280, d_out)
  network
}

#' @export
replace_head.resnet18 = function(network, d_out) {
  network$fc = nn_linear(512, d_out)
}

#' @export
replace_head.resnet34 = function(network, d_out) {
  network$fc = nn_linear(512, d_out)
}

#' @export
replace_head.resnet50 = function(network, d_out) {
  network$fc = nn_linear(512 * 4, d_out)
}

#' @export
replace_head.resnet101 = function(network, d_out) {
  network$fc = nn_linear(512 * 4, d_out)
}
#' @export
replace_head.resnet152 = function(network, d_out) {
  network$fc = nn_linear(512 * 4, d_out)
}

#' @export
replace_head.resnet101_32x8d = function(network, d_out) {
  network$fc = nn_linear(512 * 4, d_out)
}
#' @export
replace_head.resnet50_32x4d = function(network, d_out) {
  network$fc = nn_linear(512 * 4, d_out)
}

#' @export
replace_head.vgg11 = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vgg11_bn = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vgg13 = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vgg13_bn = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vgg16 = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vgg16_bn = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vgg19 = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @export
replace_head.vgg19_bn = function(network, d_out) {
  network$classifier$`6` = nn_linear(4096, d_out)
  network
}

#' @title Torchvision Learners
#' @name mlr_learners.torchvision
#' @include LearnerTorch.R
#' @description
#' Image learners from `torchvision`.
#' @section Parameters:
#' Parameters from [`LearnerTorchImage`] and
#'
#' * `pretrained` :: `logical(1)`\cr
#'   Whether to use the pretrained model.
#'
#' @export
NULL

#' @title AlexNet Image Classifier
#'
#' @templateVar name alexnet
#' @templateVar task_types classif
#' @template learner
#' @template params_learner
#'
#' @description
#' Historic convolutional neural network for image classification.
#'
#'
#' @references `r format_bib("krizhevsky2017imagenet")`
#' @include LearnerTorchImage.R
#' @rdname
#' @export
LearnerTorchAlexNet = R6Class("LearnerTorchAlexNet",

  inherit = LearnerTorchImage,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      param_set = ps(
        pretrained = p_lgl(tags = c("required", "train"))
      )
      param_set$values = list(pretrained = TRUE)
      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".alexnet"),
        param_set = param_set,
        man = "mlr3torch::mlr_learners.alexnet",
        optimizer = optimizer,
        loss = loss,
        callbacks = callbacks,
        label = "AlexNet Image Classifier"
      )
    }
  ),
  private = list(
    .network = function(task, param_vals) {
      nout = get_nout(task)
      if (param_vals$pretrained) {
        network = torchvision::model_alexnet(pretrained = TRUE)

        network$classifier$`6` = torch::nn_linear(
          in_features = network$classifier$`6`$in_features,
          out_features = nout,
          bias = TRUE
        )
        return(network)
      }

      torchvision::model_alexnet(pretrained = FALSE, num_classes = nout)
    }
  )
)

#' @include zzz.R
register_learner("classif.alexnet", function() {
  LearnerTorchVision$new("alexnet", torchvision::model_alexnet, "AlexNet", 1000L)
})

register_learner("classif.inception_v3", function() {
  LearnerTorchVision$new("inception_v3", torchvision::model_inception_v3, "Inception V3", 1000L)
})

register_learner("classif.mobilenet_v2", function() {
  LearnerTorchVision$new("mobilenet_v2", torchvision::model_mobilenet_v2, "Mobilenet V2", 1000L)
})

register_learner("classif.resnet18", function() {
  LearnerTorchVision$new("resnet18", torchvision::model_resnet18, "ResNet 18", 1000L)
})

register_learner("classif.resnet34", function() {
  LearnerTorchVision$new("resnet34", torchvision::model_resnet34, "ResNet 34", 1000L)
})

register_learner("classif.resnet50", function() {
  LearnerTorchVision$new("resnet50", torchvision::model_resnet50, "ResNet 50", 1000L)
})

register_learner("classif.resnet101", function() {
  LearnerTorchVision$new("resnet101", torchvision::model_resnet101, "ResNet 101", 1000L)
})

register_learner("classif.resnet152", function() {
  LearnerTorchVision$new("resnet152", torchvision::model_resnet152, "ResNet 152", 1000L)
})

register_learner("classif.resnet101_32x8d", function() {
  LearnerTorchVision$new("resnet101_32x8d", torchvision::model_resnet101_32x8d, "ResNet 101_32x8d", 1000L)
})

register_learner("classif.resnet50_32x4d", function() {
  LearnerTorchVision$new("resnet50_32x4d", torchvision::model_resnet50_32x4d, "ResNet 50_32x4d", 1000L)
})

register_learner("classif.vgg11", function() {
  LearnerTorchVision$new("vgg11", torchvision::model_vgg11, "VGG 11", 1000L)
})

register_learner("classif.vgg11_bn", function() {
  LearnerTorchVision$new("vgg11_bn", torchvision::model_vgg11_bn, "VGG 11", 1000L)
})

register_learner("classif.vgg13", function() {
  LearnerTorchVision$new("vgg13", torchvision::model_vgg13, "VGG 13", 1000L)
})

register_learner("classif.vgg13_bn", function() {
  LearnerTorchVision$new("vgg13_bn", torchvision::model_vgg13_bn, "VGG 13", 1000L)
})

register_learner("classif.vgg16", function() {
  LearnerTorchVision$new("vgg16", torchvision::model_vgg16, "VGG 16", 1000L)
})

register_learner("classif.vgg16_bn", function() {
  LearnerTorchVision$new("vgg16_bn", torchvision::model_vgg16_bn, "VGG 16", 1000L)
})

register_learner("classif.vgg19", function() {
  LearnerTorchVision$new("vgg19", torchvision::model_vgg19, "VGG 19", 1000L)
})

register_learner("classif.vgg19_bn", function() {
  LearnerTorchVision$new("vgg19_bn", torchvision::model_vgg19_bn, "VGG 19", 1000L)
})
