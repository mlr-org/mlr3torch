
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

*   [x] CallbackSetProgress
*   [x] CallbackSetHistory
*   [x] torch_callback
*   [x] CallbackSet
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
*   [x] as_torch_optimizer.TorchLoss
*   [x] as_torch_optimizer.nn_loss
*   [x] as_torch_optimizer
*   [x] t_loss
*   [x] mlr3torch_losses
*   [x] as_torch_optimizer.character

**Other**

*   [x] argument_matcher
*   [ ] expect_pipeop_torch
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


Some notes:

* Should we enforce the "train" tag for Torch{Optimizer, Loss, Callback}, i.e.
  set it in the initialize method of LearnerTorch?
* [ ] in learner construction ensure usage of "train" and "predict" parameters
* [ ] Utility functions for save_torch_learner and load_torch_learner, this should also be called in callback checkpoint
* Custom Backend for torch datases with hardcoded metadata, torchdatasets in suggests and not depends
* [ ] Fix the dictionary issue for the learners and the tasks
* ctx in intitiaalize self zuweisen: Dann ist klar dass das immer das gleiche Objekt ist
* rename TorchLoss, TorchOptimizer, TorchCallback to e.g. TorchDescriptorLoss
* images like in torchvision (not in inst but in tests)
* assignment for imageuri
* nano imagenet is pointless task, only one image per class.
  Rather use only two classes and distinguish between them in tests.
* Conflict between lgr and progress
* investigate caching of networks and datasets
* Ensure that the names of the output of the dataloader correspond to the names of the network's forward function
  Beware ... inputs!
  Needs renaming some existing dataloaders and an informative error message
* What happens if there are two ingress tokens but one of them returns an empty tensor?
* add saving to callback checkpoint
  I.e. the dataset_num, dataset_img etc. need to have an argument "argname" or something
* For tasks: Get a better task than nano_imagenet. Ideally one with images of different shape.
  Then remove it from the
* cloning of pipeopmodule --> adapt expect_deep_clone for torch
* warn when cloning custom callbacks
* tests with non-standard row ids
* test all parameters from paramset_torchlearner (in the learner)
* What exactly does torch_num_set_threads do? What exactly does it influence?
  Also note that the documentation says that it cannot be set on Mac
* Maybe with deep learning code we should not only test on ubuntu, but also on windows and mac?
* Create a with_xxx function that sets the torch seed, the normal seed and torch threads and unsets them afterwards
* [ ] Some image tasks
* Why did the initialize method with param_set = self$param_set work???
* Use with_torch_manual_seed when available
* save_ctx as callback for testing
* How to clone LearnerTorchModel (When .network is the trained network (?), what happens when calling train twice ???)
  --> Probably this learner should NOT be exported or it must be clearly documented how this learner behaves.
  We could address this by cloning but this would cost a lot of performance in every train call of a torch graph learner.
  Alternatively, the PipeOpTorchXXX objects could store the call to create the nn_module() instead of creating the nn_module()
  immediately.
* What about the learner that takes in a module?


**Other**
*   Check how the output names are generated, when outputs of non-terminal nodes are used
    in "output_map" ("output_<id>_output.<channel>")

* [ ] Implement the torch methods with explicit parameters in the function so that we can better check whether a parameter
from paramset_torchlearner is actually doing something

**Test Coverage**
* run paramtest vs expect_paramset --> decide for one and delete the other
* [ ] Proper testing for torch learners:
      * don't need the autotest as we don't really test the learners.
      * Instead we need to that:
        * network is generated correctly from the parameters
        * deep cloning works
        * The callbacks, optimizer, and loss are correctly set
        * the dataloader generates the data correctly and matches the network structure.
          This includes checking that the parameters (device, batch_size and shuffle are used correctly).
          Need to be extendinble to future parameters like num_workers
* [ ] Check that defaults and initial values are correctly used everywhere
* [ ] Autotest should check that all parameters are tagged with train and predict etc. Generally determine usage of tags.
* [ ] Run some tests on gpu
* [ ] ensure proper use of tags in e.g.  `param_set$get_values(tags = "train")` in expect_learner_torch
* use meta device to test device placement
* What happens if a pipeoptorch ingress gets 0 features?
* test: Manual test: Classification and Regression, test that it works with selecting only one features ("mpg") (is currently a bug)
* add test for predict_newdata
* name variables and test better
* [ ] properly test the shapes_out() method of the pipeoptorch
* [ ] Parameters must have default or tag required (?)
* [ ] Check that mlr_pipeops can still be converted to dict
* [ ] autotest for torch learner should ensure that optimizer and loss can be set in construction
* [ ] Test that the defaults of the activation functions are correctly implemented
* [ ] Properly refactor the test helpers (classes and modules etc) in other files and dont keep them in the tests.
* [ ] Use the tests from mlr3pipelines for all the pipeops
* [ ] Test the updated versions of the Descriptors
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
* [ ] remove unneeded test helper functions from mlr3pipelines
* Rename the TorchLoss, TorchOptimizer, TorchCallback to TorchCallback, TorchOptimizer, TorchLoss.
* mention that .dataset() must take row ids into account and add tests with reverse row ids for iris, mtcars, nano_imagenet, ...
* expect_learner_torch should test all private methods (like .network, .dataloader).
  Because all the learners are essentially the same (except for these methods) autotests don't make sense.
  Instead we test the general training funtions once and then the .network and .dataloader for each object (and that they match each other, e.g. with respect to argument names of the network's forward function)
* Must be possible to randomly initialize a seed. This is desireable when resampling a nn as one might want to take
  the variance of the initialization into account (currently all resamplings would be run with the same initialization)
* LearnerTorchModel:
  * Deep cloning of learner torch model is problematic.
  * calling `$train()` twice keeps training the network which we probably don't want.
* document somewhere that MPS device is not reprodicle with with_torch_manual_seed
* Use lgr instead of cat etc., in general do a lot more logging


**Maybe**

**Documentation**

* Vignette



**Once everything above is stable**

* How large are the resulting objects? We potentially store a whole lot of R6 classes in one learner.
  This might slow down stuff like batchtools CONSIDERABLY
* use logging properly
* [ ] Cloning of trained networks (requires new torch version)
* [ ] Implement bundling
* Possibility to keep the last validation prediction when doing train-predict to avoid doing this twice
* create logo for package
* [ ] Setup benchmark scripts that also run on GPU and run them at least once
* [ ] Check which versions of the dependencies we require
* create deep learning pipeop block called "paragraph"
* add references to pkgdown website
* Implement parameters "" and "early_stopping_rounds" and other parameters
* [ ] general method for freezing and unfreezing parameters.
* [ ] support the `weights` property for the learners.
* [ ] Calling `benchmark()` and evaluate the jobs on different GPUs?
* [ ] Check overhead on cpu and small batch sizes
* [ ] Minimize the time the tests run!

* [ ] Reset layer things
* freezing as a callback
* maybe integrate reset_last_layer into image learner, add property of learner that is something like "pretrained"
* [ ] The tabnet learner (we then only have the mlp learner and tabnet but should be enough for the beginning)
* [ ] The image learners
* lr scheduler as callback
* tensorboard callback
* [ ] Add the learners from the attic and all image learners
* [ ] Create {classif, regr}.torch_module learner to create custom torch learners (classif.torch did not really work because of the dataloader)
* [ ] Implement early stopping

Advertisement:

* [ ] The torch website features packages that build on top of torch
* [ ] Maybe we can write a blogpost for the RStudio AI blog?

Martin:
* Names of network and ingress token in the case where there the `x` element of the dataloader has length 1?

