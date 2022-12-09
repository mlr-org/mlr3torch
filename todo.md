**PipeOps**

* [x] PipeOpModule

* [x] PipeOpTorchIngress
* [x] PipeOpTorchIngressNumeric
* [x] PipeOpTorchIngressImage
* [x] PipeOpTorchIngressCategorical

* [x] PipeOpTorchConv
* [x] PipeOpTorchConv1D
* [x] PipeOpTorchConv2D
* [x] PipeOpTorchConv3D

* [x] PipeOpTorchConvTranspose1D
* [x] PipeOpTorchConvTranspose2D
* [x] PipeOpTorchConvTranspose3D
* [x] PipeOpTorchConvTranspose

* [x] PipeOpTorchUnsqueeze
* [x] PipeOpTorchSqueeze
* [x] PipeOpTorchFlatten
* [x] PipeOpTorchReshape

* [x] PipeOpTorchBatchNorm
* [x] PipeOpTorchBatchNorm1D
* [x] PipeOpTorchBatchNorm2D
* [x] PipeOpTorchBatchNorm3D

* [x] PipeOpTorchAvgPool
* [x] PipeOpTorchAvgPool1D
* [x] PipeOpTorchAvgPool2D
* [x] PipeOpTorchAvgPool3D

* [ ] PipeOpTorchMaxPool
* [ ] PipeOpTorchMaxPool1D
* [ ] PipeOpTorchMaxPool2D
* [ ] PipeOpTorchMaxPool3D

* [ ] PipeOpTorchLinear
* [ ] PipeOpTorchHead

* [ ] PipeOpTorchMerge
* [ ] PipeOpTorchMergeSum
* [ ] PipeOpTorchMergeProd
* [ ] PipeOpTorchMergeCat

* [ ] PipeOpTorchDropout
* [ ] PipeOpTorchOptimizer

* [ ] PipeOpTorchLoss

* [ ] PipeOpTorchLogSigmoid
* [ ] PipeOpTorchSigmoid
* [ ] PipeOpTorchGELU
* [ ] PipeOpTorchReLU
* [ ] PipeOpTorchActTanhShrink
* [ ] PipeOpTorchGLU
* [ ] PipeOpTorchCelu
* [ ] PipeOpTorchThreshold
* [ ] PipeOpTorchRReLU
* [ ] PipeOpTorchLayerNorm
* [ ] PipeOpTorchHardSigmoid
* [ ] PipeOpTorchPReLU
* [ ] PipeOpTorchActTanh
* [ ] PipeOpTorchLeakyReLU
* [ ] PipeOpTorchRelu6
* [ ] PipeOpTorchELU
* [ ] PipeOpTorchtSoftShrink
* [ ] PipeOpTorchHardShrink
* [ ] PipeOpTorchSoftPlus
* [ ] PipeOpTorchSELU
* [ ] PipeOpTorchSoftmax
* [ ] PipeOpTorchActSoftSign
* [ ] PipeOpTorchHardTanh

* [ ] PipeOpTorchModel
* [ ] PipeOpTorchModelRegr
* [ ] PipeOpTorchModelClassif

* [ ] PipeOpTorch

**Learner**

* [ ] LearnerClassifTorchModel
* [ ] LearnerClassifTorchAbstract
* [ ] LearnerClassifTorch
* [ ] paramset_torchlearner
* [ ] train_loop
* [ ] torch_network_predict
* [ ] torch_network_train
* [ ] encode_prediction
* [ ] learner_torch_train
* [ ] learner_torch_predict

**Callbacks**

* [ ] CallbackTorchProgress
* [ ] CallbackTorchHistory
* [ ] callback_torch
* [ ] CallbackTorch
* [ ] ContextTorch
* [ ] t_clbk

**Graph**

* [ ] nn_graph
* [ ] ModelDescriptior
* [ ] print.ModelDescriptor
* [ ] model_descriptor_to_module
* [ ] model_descriptor_union
* [x] print.TorchIngressToken
* [x] batchgetter_categ
* [x] TorchIngressToken

**Optimizer**

* [ ] TorchOptimizer
* [ ] as_torch_optimizer
* [ ] as_torch_optimizer.character
* [ ] mlr3torch_optimizers
* [ ] as_torch_optimizer.TorchOptimizer
* [ ] as_torch_optimizer.torch_optimizer_generator
* [ ] t_opt

**Loss**

* [ ] TorchLoss
* [ ] as_torch_loss.TorchLoss
* [ ] as_torch_loss.nn_loss
* [ ] as_torch_loss
* [ ] t_loss
* [ ] mlr3torch_losses
* [ ] as_torch_loss.character


**Other**

* [ ] argument_matcher
* [ ] check_network
* [ ] autotest_torchop
* [ ] avg_output_shape
* [ ] po_register_env
* [ ] Tiny Imagenet
* [x] pots --> removed
* [x] pot --> removed
* [ ] inferps
* [ ] register_po
* [ ] check_measures
* [ ] make_activation
* [ ] register_mlr3
* [ ] check_vector
* [ ] conv_transpose_output_shape
* [ ] conv_output_shape
* [ ] lg
* [ ] expect_torchop
* [ ] check_callbacks
* [ ] load_task_tiny_imagenet
* [ ] batchgetter_num
* [ ] task_dataset
* [ ] unique_id
* [ ] measure_prediction
* [ ] toytask
* [ ] register_mlr3pipelines
* [x] imageuri


**NNs**

* [ ] nn_merge_prod
* [ ] nn_unsqueeze
* [ ] nn_merge_cat
* [ ] nn_squeeze
* [ ] nn_merge_sum
* [ ] nn_reshape

**Roxy**

* [ ] roxy_pipeop_torch_fields_default
* [ ] roxy_pipeop_torch_construction
* [ ] roxy_param_module_generator
* [ ] roxy_pipeop_torch_license

* [ ] roxy_param_innum
* [ ] roxy_param_packages
* [ ] roxy_pipeop_torch_format
* [ ] roxy_param_id
* [ ] roxy_param_param_vals
* [ ] roxy_pipeop_torch_channels_default
* [ ] roxy_param_param_set
* [ ] roxy_pipeop_torch_param_id
* [ ] roxy_pipeop_torch_state_default
* [ ] roxy_construction
* [ ] roxy_pipeop_torch_methods_default




# Other

* early_stopping -> test renaming
* ensure that caching does what we want the caching to do (tiny imagenet)

**Mit Martin**

* Rename PipeOpTorch -> PipeOpNN: Names would better represent the class hierarchy.
* It must be documented how the output names are generated, when outputs of non-terminal nodes are used
in "output_map" ("output_<id>_output.<channel>")

