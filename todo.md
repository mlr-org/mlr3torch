**PipeOps**

*   [x] PipeOpModule

*   [x] PipeOpTorchMerge

*   [x] PipeOpTorchMergeSum

*   [x] PipeOpTorchMergeProd

*   [x] PipeOpTorchMergeCat

*   [x] PipeOpTorchIngress

*   [x] PipeOpTorchIngressNumeric

*   [x] PipeOpTorchIngressImage

*   [x] PipeOpTorchIngressCategorical

*   [x] PipeOpTorchConv

*   [x] PipeOpTorchConv1D

*   [x] PipeOpTorchConv2D

*   [x] PipeOpTorchConv3D

*   [x] PipeOpTorchConvTranspose1D

*   [x] PipeOpTorchConvTranspose2D

*   [x] PipeOpTorchConvTranspose3D

*   [x] PipeOpTorchConvTranspose

*   [x] PipeOpTorchUnsqueeze

*   [x] PipeOpTorchSqueeze

*   [x] PipeOpTorchFlatten

*   [x] PipeOpTorchReshape

*   [x] PipeOpTorchBatchNorm

*   [x] PipeOpTorchBatchNorm1D

*   [x] PipeOpTorchBatchNorm2D

*   [x] PipeOpTorchBatchNorm3D

*   [x] PipeOpTorchAvgPool

*   [x] PipeOpTorchAvgPool1D

*   [x] PipeOpTorchAvgPool2D

*   [x] PipeOpTorchAvgPool3D

*   [x] PipeOpTorchMaxPool

*   [x] PipeOpTorchMaxPool1D

*   [x] PipeOpTorchMaxPool2D

*   [x] PipeOpTorchMaxPool3D

*   [x] PipeOpTorchLinear

*   [x] PipeOpTorchHead

*   [x] PipeOpTorchDropout

*   [x] PipeOpTorchLogSigmoid

*   [x] PipeOpTorchSigmoid

*   [x] PipeOpTorchGELU

*   [x] PipeOpTorchReLU

*   [x] PipeOpTorchActTanhShrink

*   [x] PipeOpTorchGLU

*   [x] PipeOpTorchCelu

*   [x] PipeOpTorchThreshold

*   [x] PipeOpTorchRReLU

*   [x] PipeOpTorchHardSigmoid

*   [x] PipeOpTorchPReLU

*   [x] PipeOpTorchActTanh

*   [x] PipeOpTorchLeakyReLU

*   [x] PipeOpTorchRelu6

*   [x] PipeOpTorchELU

*   [x] PipeOpTorchtSoftShrink

*   [x] PipeOpTorchHardShrink

*   [x] PipeOpTorchSoftPlus

*   [x] PipeOpTorchSELU

*   [x] PipeOpTorchSoftmax

*   [x] PipeOpTorchActSoftSign

*   [x] PipeOpTorchHardTanh

*   [x] PipeOpTorchLayerNorm

*   [x] PipeOpTorchOptimizer

*   [x] PipeOpTorchLoss

*   [x] PipeOpTorchModel

*   [x] PipeOpTorchModelRegr

*   [x] PipeOpTorchModelClassif

**Learner**

*   [x] LearnerClassifTorchModel
*   [x] LearnerClassifTorchAbstract
*   [x] paramset_torchlearner
*   [o] train_loop (not sure how to test this)
*   [x] torch_network_predict
*   [x] encode_prediction
*   [o] learner_torch_train (not sure how to test this)
*   [o] learner_torch_predict (not sure how to test this)

**Callbacks**

*   [x] CallbackTorchProgress
*   [x] CallbackTorchHistory
*   [x] callback_torch
*   [x] CallbackTorch
*   [x] ContextTorch
*   [x] t_clbk

**Graph**

*   [ ] PipeOpTorch
*   [x] nn_graph
*   [x] ModelDescriptior
*   [x] print.ModelDescriptor
*   [x] model_descriptor_to_module
*   [x] model_descriptor_to_learner
*   [x] model_descriptor_union
*   [x] print.TorchIngressToken
*   [x] batchgetter_categ
*   [x] TorchIngressToken

**Optimizer**

*   [x] TorchOptimizer
*   [x] as_torch_optimizer
*   [x] as_torch_optimizer.character
*   [x] mlr3torch_optimizers
*   [x] as_torch_optimizer.TorchOptimizer
*   [x] as_torch_optimizer.torch_optimizer_generator
*   [x] t_opt

**Loss**

*   [x] TorchLoss
*   [x] as_torch_loss.TorchLoss
*   [x] as_torch_loss.nn_loss
*   [x] as_torch_loss
*   [x] t_loss
*   [x] mlr3torch_losses
*   [x] as_torch_loss.character

**Other**

*   [x] argument_matcher
*   [ ] check_network
*   [ ] autotest_pipeop_torch
*   [ ] avg_output_shape
*   [ ] po_register_env
*   [ ] Tiny Imagenet
*   [ ] inferps
*   [ ] register_po
*   [ ] check_measures
*   [ ] make_activation
*   [ ] register_mlr3
*   [ ] check_vector
*   [ ] conv_transpose_output_shape
*   [ ] conv_output_shape
*   [ ] lg
*   [ ] expect_torchop
*   [ ] check_callbacks
*   [ ] load_task_tiny_imagenet
*   [ ] batchgetter_num
*   [ ] task_dataset
*   [x] unique_id
*   [ ] measure_prediction
*   [ ] toytask -> this name sucks
*   [ ] register_mlr3pipelines
*   [x] imageuri

**NNs**

*   [ ] nn_merge_prod
*   [ ] nn_unsqueeze
*   [ ] nn_merge_cat
*   [ ] nn_squeeze
*   [ ] nn_merge_sum
*   [ ] nn_reshape

**Roxy**

TODO: This list is not up to date anymore

*   [ ] roxy_pipeop_torch_fields_default


*   [ ] roxy_param_module_generator

*   [ ] roxy_torch_license

*   [ ] roxy_param_innum

*   [ ] roxy_param_packages

*   [ ] roxy_pipeop_torch_format

*   [ ] roxy_param_id

*   [ ] roxy_param_param_vals

*   [ ] roxy_pipeop_torch_channels_default

*   [ ] roxy_param_param_set

*   [ ] roxy_pipeop_torch_param_id

*   [ ] roxy_pipeop_torch_state_default

*   [ ] roxy_construction

*   [ ] roxy_pipeop_torch_methods_default


All the learner implementations: 

* [ ] LearnerClassifMLP
* [ ] Image Learners: They are so similar, we might create them all programmatically with one help page.

# Other


**Important**

* [ ] Cloning of trained networks (requires new torch version)
* [ ] Reproducibility: Add the cuda seed resetting
* [ ] Implement bundling
* [ ] Add the learners etc. for regression

**Missing stuff**

* [ ] The image learners
* [ ] Some image tasks
* [ ] The tabnet learner (we then only have the mlp learner and tabnet but should be enough for the beginning)

**Refactors**

* [ ] We should structure the parameters better with tags to define which function gets what 
* [ ] Add the learners from the attic and all image learners
* [ ] Use meta device in tests wherever possible to make tests run as fast as possible.
* [ ] ensure that caching does what we want the caching to do (tiny imagenet)
* [ ] ensure proper use of tags in e.g.  `param_set$get_values(tags = "train")`
* [ ] Autotest should check that all parameters are tagged with train and predict etc. Generally determine usage of tags.
* [ ] Run some tests on gpu
* [x] Rename Debug Torch Learner to featureless and export

**Other**
* [ ] Check that defaults and initial values are correctly used everywhere
* [ ] Is withr important anyway? If yes, then remove the with_seed function, otherwise remove withr from imports
* [ ] Reset the torch seed after ending the `$train()` call of the learner.
* [ ] Check which versions of the packages we actually require
* [ ] Check which man files are actually used and remove the rest

* [ ] Implement the torch methods with explicit parameters in the function so that we can better check whether a parameter 
from paramset_torchlearner is actually doing something

**Consistency**
* [ ] Check that all the mlr3torch_activations are simple and maybe rename to activations_simple. 
Also add tests or sth. (For learners that allow to set the activation function but expect it to be a scalar). 

**Performance**


* [ ] Setup benchmark scripts that also run on GPU and run them at least once

**Test Coverage**
* Parameters must have default or tag required (?)
* [ ] Check that mlr_pipeops can still be converted to dict
* [ ] autotest for torch learner should ensure that optimizer and loss can be set in construction
* [ ] Test that the defaults of the activation functions are correctly implemented
* [ ] Properly refactor the test helpers (classes and modules etc) in other files and dont keep them in the tests.
* [ ] Use the tests from mlr3pipelines for all the pipeops
* [ ] Test the updated versions of the TorchWrapper
* [ ] Deep clones of torch modules: 
-> the function that checks for deep clones needs to skip some torch-specific stuff, e.g. the attribute "module" 
for nn modules, or "Optimizer" for optim_adam etc.
* [ ] Test that the default values of the pipeops are correctly documented
* Write expect_learner_torch that checks all the properties a torch learner has to satisfy
* [ ] Meta tests for the functions / objects created for the tests (like PipeOpTorchDebug)
* [ ] Parameter tests for callbacks

**Cosmetic**
* [x] Better printer for ModelDescriptor (see whether loss is configured e.g.)
* [x] Rename LearnerClassifTorchAbstract to LearnerClassifTorch and LearnerClassifTorch to LearnerClassifTorchModule

**Documentation**

* [ ] Define the @family tags somewhere and then annotate everything correctly
* [ ] Use the `tags` constrctor argument from PipeOps
* [ ] Check that objects have the correct families in the documentation.
* [ ] Add tests for documentation, i.e. that all the required sections are present
* [ ] Write test that all construction arguments are documented.
* [ ] Properly document that the classifiers must return the scores and no the probabilities
* Add examples for all PipeOps (maybe a template for PipeOpTorch?)
*   It must be documented how the output names are generated, when outputs of non-terminal nodes are used
    in "output_map" ("output_<id>_output.<channel>")
* [ ] Make the paramset torch leaner template a roxygen function, it should be directly seen for every learner (?).

In the future (soon): 

* [ ] Create {classif, regr}.torch_module learner to create custom torch learners (classif.torch did not really work because of the dataloader)
* [ ] Maybe it should be possible to easily overwrite the dataloader for a learner (?) 
* [ ] Implement early stopping and all other parameters from paramset torchlearner. 

**In the future**

* [ ] general method for freezing and unfreezing parameters.
* [ ] support the `weights` property for the learners.
* [ ] Calling `benchmark()` and evaluate the jobs on different GPUs?
* [ ] Check overhead on cpu and small batch sizes


**Optimization**

* [ ] Minimize the time the tests run!
  (utilize the fetureless torch learner as much as possible, should probbably extend it to all feature types)
