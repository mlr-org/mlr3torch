* [ ] in learner construction ensure usage of "train" and "predict" parameters
* Conflict between lgr and progress
* What happens if there are two ingress tokens but one of them returns an empty tensor?
* tests with non-standard row ids
* test all parameters from paramset_torchlearner (in the learner)
*   Check how the output names are generated, when outputs of non-terminal nodes are used
    in "output_map" ("output_<id>_output.<channel>")

**Test Coverage**
* [ ] Proper testing for torch learners:
      * don't need the autotest as we don't really test the learners.
      * Instead we need to that:
        * network is generated correctly from the parameters
        * deep cloning works
        * The callbacks, optimizer, and loss are correctly set
        * the dataloader generates the data correctly and matches the network structure.
          This includes checking that the parameters (device, batch_size and shuffle are used correctly).
          Need to be extendinble to future parameters like num_workers
* [ ] Autotest should check that all parameters are tagged with train and predict etc. Generally determine usage of tags.
* [ ] Run some tests on gpu
* [ ] ensure proper use of tags in e.g.  `param_set$get_values(tags = "train")` in expect_learner_torch
* use meta device to test device placement
* What happens if a pipeoptorch ingress gets 0 features?
* test: Manual test: Classification and Regression, test that it works with selecting only one features ("mpg") (is currently a bug)
* [ ] properly test the shapes_out() method of the pipeoptorch
* [ ] autotest for torch learner should ensure that optimizer and loss can be set in construction
* [ ] Test that the defaults of the activation functions are correctly implemented
* [ ] Properly refactor the test helpers (classes and modules etc) in other files and dont keep them in the tests.
* [ ] Use the tests from mlr3pipelines for all the pipeops
* [ ] Test that the default values of the pipeops are correctly documented
* Write expect_learner_torch that checks all the properties a torch learner has to satisfy
* [ ] Meta tests for the functions / objects created for the tests (like PipeOpTorchDebug)
