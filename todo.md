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
*   [ ] train_loop
*   [ ] torch_network_predict
*   [ ] torch_network_train
*   [ ] encode_prediction
*   [ ] learner_torch_train
*   [ ] learner_torch_predict

**Callbacks**

*   [ ] CallbackTorchProgress
*   [ ] CallbackTorchHistory
*   [ ] callback_torch
*   [ ] CallbackTorch
*   [ ] ContextTorch
*   [ ] t_clbk

**Graph**

*   [ ] PipeOpTorch
*   [ ] nn_graph
*   [ ] ModelDescriptior
*   [ ] print.ModelDescriptor
*   [ ] model_descriptor_to_module
*   [ ] model_descriptor_union
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

*   [ ] argument_matcher
*   [ ] check_network
*   [ ] autotest_torchop
*   [ ] avg_output_shape
*   [ ] po_register_env
*   [ ] Tiny Imagenet
*   [x] pots --> removed
*   [x] pot --> removed
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
*   [ ] unique_id
*   [ ] measure_prediction
*   [ ] toytask
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

*   [ ] roxy_pipeop_torch_fields_default

*   [ ] roxy_pipeop_torch_construction

*   [ ] roxy_param_module_generator

*   [ ] roxy_pipeop_torch_license

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

# Other

* Add the stuff for regression 

* Add the learners from the attic and all image learners

* ensure that the functions like .network and .dataloader always get the parameter values with the defautl set ->
Maybe they need to be called by a wrapper .get_network()

*   early_stopping -> test renaming

*   Use meta device in tests wherever possible to make tests run as fast as possible.

*   ensure that caching does what we want the caching to do (tiny imagenet)

*   ensure proper use of tags in e.g.  `param_set$get_values(tags = "train")`

*   Rename PipeOpTorch -> PipeOpNN: Names would better represent the class hierarchy:
    note that all the roxygen stuff has to be renamed too and adjusted.
    Maybe PipeOpTorch -> PipeOpTorchNN because we still want to refer to all the PipeOpTorch's

*   Check that deep clones for callbacks work

*   what happens to the logs during hotstarting

*   Add hotstarting everywhere

*   Autotest should check that all parameters are tagged with train and predict etc.

*   Ensure that one cannot change parameters inbetween training and predictino and break stuff in weird ways
    --> properly tagging parameters and properly define learner methods.

* general method for freezing and unfreezing parameters.

* I think the private method `.dataloader` should call the `.dataset` method and only the latter has to be implemented, 
    as the `.dataloader()` call should be mostly identical in almost all cases. Of course it can still be overwritten. 
    By doing so we also habe more control, that the shuffle argument is respected. Users can always overwrite the 
    .dataloader metod as well if they want to.

* We should structure the parameters better with tags to define which function gets what 

* If the callbacks get parameters, these should probably be added to the parameter set of the torch learner as well, 
    e.g. cb.scheduler.param. We must then add a check that no parameter starts with cb and extend the tests with 
    respect to the deep cloning etc.

* support the `weights` property for the learners.

* The history callback should always be set and we should have a standardized accessor for the history.

* Run some tests on gpu

* Setup benchmark script

* Better printer for ModelDescriptor (see whether loss is configured e.g.)

* Check that defaults and initial values are correctly used eveywhere

* Test that the default values of the pipeops are correctly documented

* Use the tests from mlr3pipelines for all the pipeops

* Check that mlr_pipeops can still be converted to dict

* Add tests for documentation, i.e. that all the required sections are present

* Deep clones of torch modules: 
-> the function that checks for deep clones needs to skip some torch-specific stuff, e.g. the attribute "module" 
for nn modules, or "Optimizer" for optim_adam etc.

Properly refactor the test helpers (classes and modules etc) in other files and dont keep them in the tests.

* Properlu document that the classifiers must return the scores

* Implement the torch methods with explicit parameters in the function so that we can better check whether a parameter 
from paramset_torchlearner is actually doing something

* Implement early stopping and all other parameters from paramset torchlearner

* Callback history should always be added and there should be quick accessors for hist_valid and hist_train in the 
torch network

* Better solution for callbacks than the one with the state. This sucks

* Maybe the callbacks should also be a construction argument

* Minimize the time the tests run!

* convenience function for callbacks

* t_clbk should actually return a class. But what do we do with the initialization arguments? 
The current implementation sucks, because when we train a learner twice, all the callbacks are already trained to 
the callbacks should not be executable but because this check is only performed when setting the parameters the error
is not caught:

What we want: 

All the persistent information of a callback is stored in its state. Before training begins, the state of the callback is reset.

* Add a parameter measures that allows to set measures_train and measures_valid simultanteously

* Make the paramset torch leaner template a roxygen function, it should be immediately seen at every learner.

* test that objects have the correct families in the documentation.

* DRY with respect to the dataloader

* Test that the defaults of the activation functions are correctly implemented

* Check that all the mlr3torch_activations are simple and maybe rename to activations_simple. 
Also add tests or sth.

* autotest for torch learner should ensure that optimizer and loss can be set in construction

* Write test that all construction arguments are documented.

* I think I might go to r6 = true again... it is so much trouble to do it without it and is much less organized.


**Mit Martin**

*   It must be documented how the output names are generated, when outputs of non-terminal nodes are used
    in "output_map" ("output_<id>_output.<channel>")
