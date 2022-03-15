library(mlr3torch)

# AlexNet
model <- torchvision::model_alexnet(pretrained = TRUE)
model$classifier[[7]]$out_feature

model <- reset_last_layer(model, num_classes = 10)
model$classifier[[7]]$out_feature

model <- torchvision::model_resnet18(pretrained = TRUE)
model$fc$out_feature

model <- reset_last_layer(model, num_classes = 10)
model$fc$out_feature


# Reprex maybe for torch(vision)


model_alexnet <- torchvision::model_alexnet(pretrained = TRUE)

model_alexnet

model_alexnet$classifier

model_alexnet$classifier[[7]]

model_alexnet$classifier$`6` <- torch::nn_linear(4096, 10)

model_alexnet$classifier[[7]]

model_alexnet$classifier

# Works as intended
model_alexnet$classifier$`6` <- torch::nn_linear(4096, 10)
