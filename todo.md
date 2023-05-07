
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


All the learner implementations: 

* [ ] LearnerClassifMLP
* [ ] Image Learners: They are so similar, we might create them all programmatically with one help page.

# Other


**Important**

* [ ] Add the learners etc. for regression
* [ ] Cloning of trained networks (requires new torch version)
* [ ] Reproducibility: Add the cuda seed resetting
* [ ] param_set$get_values(tags = "train") expects all parameters to be tagged with "train" --> maybe this is currently a bug
* [ ] tiny_imagenet --> We can create construction argument download and then as.data.table.DictionaryTasks works (look at its code for information)
* [ ] Implement bundling
* [ ] Utility functions for save_torch_learner and load_torch_learner, this should also be called in callback checkpoint
* [ ] Fix the dictionary issue and then add those objects to dictionary that we could not add because it prohibits the as.data.table conversion

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
* [ ] Check which versions of the packages we actually require
* [ ] Check which man files are actually used and remove the rest
* [ ] Exclude the sanity tests in the learner autotest
* Incorrect use of row roles (pipelines issue)
*   Check how the output names are generated, when outputs of non-terminal nodes are used
    in "output_map" ("output_<id>_output.<channel>")

* [ ] Implement the torch methods with explicit parameters in the function so that we can better check whether a parameter 
from paramset_torchlearner is actually doing something

**Consistency**
* [ ] Check that all the mlr3torch_activations are simple and maybe rename to activations_simple. 
Also add tests or sth. (For learners that allow to set the activation function but expect it to be a scalar). 

**Performance**

* [ ] Setup benchmark scripts that also run on GPU and run them at least once

**Test Coverage**
* [ ] Reset layer things
* [ ] Parameters must have default or tag required (?)
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
* [ ] Test that all optimizers are working (closure issue for lbfgs)
* [ ] Test that all losses are working
* [ ] regr learners have mse and classif ce as default loss

**Documentation**

* README (probably want to have the tabnet learner for that)
* Vignette

* Go over the core methods and verify that no old docu is left:
  * PipeOpModule
  * PipeOpTorch
  * LearnerClassifTorch
  * LearnerRegrTorch
  * LearnerRegrTorchModel
  * LearnerClassifTorchModel
  * nn_graph
  * ModelDescriptor
  * TorchWrapper and all the subclasses


In the future (soon): 

* [ ] Create {classif, regr}.torch_module learner to create custom torch learners (classif.torch did not really work because of the dataloader)
* [ ] Maybe it should be possible to easily overwrite the dataloader for a learner (?) 
* [ ] Implement early stopping and all other parameters from paramset torchlearner. 


Advertisement:

* [ ] The torch website features packages that build on top of torch
* [ ] Maybe we can write a blogpost for the RStudio AI blog?

**In the future**

* [ ] general method for freezing and unfreezing parameters.
* [ ] support the `weights` property for the learners.
* [ ] Calling `benchmark()` and evaluate the jobs on different GPUs?
* [ ] Check overhead on cpu and small batch sizes
* [ ] Minimize the time the tests run!
  (utilize the fetureless torch learner as much as possible, should probbably extend it to all feature types)
* [ ] Simplify the documentation templates for PipeOpTorch


Fragen an Martin: 
* Wieso wird der batchgetter gecrated?
