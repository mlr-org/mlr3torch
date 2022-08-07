devtools::load_all()

# nn_graph

# LearnerClassifTorchAbstract <-- LearnerClassif
# |  --> learner_torch_train
# |      --> ContextTorch
# |      --> History "training history for a torch learner" -- why is this not a list?
# |      --> learner_torch_predict.R: encode_prediction
# |  --> learner_torch_predict
# |  --> serialize
# |- LearnerClassifTorchModel (uses a model provided during construction)
# |- LearnerClassifAlexNet  (actually uses the .network function sensibly)
# |  --> utils-models.R: reset_last_layer
# |- LearnerClassifTorch     (gets a nn_module that has a 'task' input)
# |- LearnerClassifMLP       (uses tops)
# |- LearnerClassifTabResNet (uses tops)


# notes:
# - Some special ops should probably not inherit from torchop
# - want to have lightweight wrapper around nn_-modules so that Graph$train() can

# |- [ ] TorchOpInput
# |- [X] TorchOpOutput
# |- [X] TorchOpModel
# |  |- [X] TorchOpModelClassif
# |  |- [X] TorchOpModelRegr
# |- [ ] TorchOpActivation: ....
# |  --> paramsets_activation
# |- [X] TorchOpLoss
# |  --> paramsets_loss
# |- [X] TorchOpSoftmax
# |- [ ] TorchOpMerge
# |  |- [ ] TorchOpAdd
# |  |- [ ] TorchOpMul
# |  |- [ ] TorchOpCat
# |- [X] TorchOpOptimizer
# |  --> paramsets_optim
# |- [X] TorchOpAvgPool
# |- [X] TorchOpConv
# |- [X] TorchOpMaxPool
# |- [ ] TorchOpBatchNorm
# |- [ ] TorchOpDropout
# |- [X] TorchOpFlatten
# |- [X] TorchOpLinear
# |- [ ] TorchOpConvTranspose
# |- [ ] TorchOpLayerNorm
# |- [ ] TorchOpReshape, TorchOpSqueeze, TorchOpUnsqueeze
# |- [-] TorchOpSelect
# |- [-] TorchOpRepeat
# |- [ ] TorchOpTabResNetBlocks
# |- [ ] TorchOpTabTokenizer

# GraphLearnerTorch: additional info on learning process; should probably be in the TorchLearner

# Callback:  should probably just be a function
# |- CallbackTorch
#    |- CallbackTorchLogger
#    |- CallbackTorchProgress

# make_paramset
# --> paramsets_loss
# --> paramsets_optim

## Auxiliary

## dictionaries
# mlr_torchops, top

## modules
# nn_cls
# nn_rtdl_attention.R: nn_ft_attention

## utils
# operators.R
# torch_reflections
# util_torch (external only)
# utils.R
# make_dl_from_task (external only, superfluous?)
# as_learner_torch (should probably be just as_learner.TorchOpModel{Classif,Regr}
# reset.R: reset_parameters, reset_running_stats

## Things that do not relate to the machinery itself

### dataset / data preparation things
# df_from_imagenet_dir, img_dataset
# load_task_tiny_imagenet, toytask
# TaskClassifSpiral.R: load_task_spiral
# imagauri, transform_imageuri
# make_image_dataset
# as_dataloader --> torch::dataloader()

### PipeOps
# PipeOpImageTrafo <-- PipeOpTaskPreprocSimple
# --> paramsets_image_trafo.R

### Learners
# LearnerClassifTabNet should probably be in mlr3extralearners
# LearnerClassifTabNet <-- LearnerClassif
# LearnerRegrTabnet <-- LearnerRegr
# --> params_tabnet

###########################
# Things to improve
# - LearnerClassifTorchAbstract should probably take optimizer / loss objects that contain the necessary paramset?
