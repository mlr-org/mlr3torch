**PipeOps**

*   \[x] PipeOpModule

*   \[x] PipeOpTorchMerge

*   \[x] PipeOpTorchMergeSum

*   \[x] PipeOpTorchMergeProd

*   \[x] PipeOpTorchMergeCat

*   \[x] PipeOpTorchIngress

*   \[x] PipeOpTorchIngressNumeric

*   \[x] PipeOpTorchIngressImage

*   \[x] PipeOpTorchIngressCategorical

*   \[x] PipeOpTorchConv

*   \[x] PipeOpTorchConv1D

*   \[x] PipeOpTorchConv2D

*   \[x] PipeOpTorchConv3D

*   \[x] PipeOpTorchConvTranspose1D

*   \[x] PipeOpTorchConvTranspose2D

*   \[x] PipeOpTorchConvTranspose3D

*   \[x] PipeOpTorchConvTranspose

*   \[x] PipeOpTorchUnsqueeze

*   \[x] PipeOpTorchSqueeze

*   \[x] PipeOpTorchFlatten

*   \[x] PipeOpTorchReshape

*   \[x] PipeOpTorchBatchNorm

*   \[x] PipeOpTorchBatchNorm1D

*   \[x] PipeOpTorchBatchNorm2D

*   \[x] PipeOpTorchBatchNorm3D

*   \[x] PipeOpTorchAvgPool

*   \[x] PipeOpTorchAvgPool1D

*   \[x] PipeOpTorchAvgPool2D

*   \[x] PipeOpTorchAvgPool3D

*   \[x] PipeOpTorchMaxPool

*   \[x] PipeOpTorchMaxPool1D

*   \[x] PipeOpTorchMaxPool2D

*   \[x] PipeOpTorchMaxPool3D

*   \[x] PipeOpTorchLinear

*   \[x] PipeOpTorchHead

*   \[x] PipeOpTorchDropout

*   \[x] PipeOpTorchLogSigmoid

*   \[x] PipeOpTorchSigmoid

*   \[x] PipeOpTorchGELU

*   \[x] PipeOpTorchReLU

*   \[x] PipeOpTorchActTanhShrink

*   \[x] PipeOpTorchGLU

*   \[x] PipeOpTorchCelu

*   \[x] PipeOpTorchThreshold

*   \[x] PipeOpTorchRReLU

*   \[x] PipeOpTorchHardSigmoid

*   \[x] PipeOpTorchPReLU

*   \[x] PipeOpTorchActTanh

*   \[x] PipeOpTorchLeakyReLU

*   \[x] PipeOpTorchRelu6

*   \[x] PipeOpTorchELU

*   \[x] PipeOpTorchtSoftShrink

*   \[x] PipeOpTorchHardShrink

*   \[x] PipeOpTorchSoftPlus

*   \[x] PipeOpTorchSELU

*   \[x] PipeOpTorchSoftmax

*   \[x] PipeOpTorchActSoftSign

*   \[x] PipeOpTorchHardTanh

*   \[x] PipeOpTorchLayerNorm

*   \[x] PipeOpTorchOptimizer

*   \[x] PipeOpTorchLoss

*   \[ ] PipeOpTorchModel

*   \[ ] PipeOpTorchModelRegr

*   \[ ] PipeOpTorchModelClassif

**Learner**

*   \[ ] LearnerClassifTorchModel
*   \[ ] LearnerClassifTorchAbstract
*   \[ ] LearnerClassifTorch
*   \[ ] paramset\_torchlearner
*   \[ ] train\_loop
*   \[ ] torch\_network\_predict
*   \[ ] torch\_network\_train
*   \[ ] encode\_prediction
*   \[ ] learner\_torch\_train
*   \[ ] learner\_torch\_predict

**Callbacks**

*   \[ ] CallbackTorchProgress
*   \[ ] CallbackTorchHistory
*   \[ ] callback\_torch
*   \[ ] CallbackTorch
*   \[ ] ContextTorch
*   \[ ] t\_clbk

**Graph**

*   \[ ] PipeOpTorch
*   \[ ] nn\_graph
*   \[ ] ModelDescriptior
*   \[ ] print.ModelDescriptor
*   \[ ] model\_descriptor\_to\_module
*   \[ ] model\_descriptor\_union
*   \[x] print.TorchIngressToken
*   \[x] batchgetter\_categ
*   \[x] TorchIngressToken

**Optimizer**

*   \[ ] TorchOptimizer
*   \[ ] as\_torch\_optimizer
*   \[ ] as\_torch\_optimizer.character
*   \[ ] mlr3torch\_optimizers
*   \[ ] as\_torch\_optimizer.TorchOptimizer
*   \[ ] as\_torch\_optimizer.torch\_optimizer\_generator
*   \[ ] t\_opt

**Loss**

*   \[ ] TorchLoss
*   \[ ] as\_torch\_loss.TorchLoss
*   \[ ] as\_torch\_loss.nn\_loss
*   \[ ] as\_torch\_loss
*   \[ ] t\_loss
*   \[ ] mlr3torch\_losses
*   \[ ] as\_torch\_loss.character

**Other**

*   \[ ] argument\_matcher
*   \[ ] check\_network
*   \[ ] autotest\_torchop
*   \[ ] avg\_output\_shape
*   \[ ] po\_register\_env
*   \[ ] Tiny Imagenet
*   \[x] pots --> removed
*   \[x] pot --> removed
*   \[ ] inferps
*   \[ ] register\_po
*   \[ ] check\_measures
*   \[ ] make\_activation
*   \[ ] register\_mlr3
*   \[ ] check\_vector
*   \[ ] conv\_transpose\_output\_shape
*   \[ ] conv\_output\_shape
*   \[ ] lg
*   \[ ] expect\_torchop
*   \[ ] check\_callbacks
*   \[ ] load\_task\_tiny\_imagenet
*   \[ ] batchgetter\_num
*   \[ ] task\_dataset
*   \[ ] unique\_id
*   \[ ] measure\_prediction
*   \[ ] toytask
*   \[ ] register\_mlr3pipelines
*   \[x] imageuri

**NNs**

*   \[ ] nn\_merge\_prod
*   \[ ] nn\_unsqueeze
*   \[ ] nn\_merge\_cat
*   \[ ] nn\_squeeze
*   \[ ] nn\_merge\_sum
*   \[ ] nn\_reshape

**Roxy**

*   \[ ] roxy\_pipeop\_torch\_fields\_default

*   \[ ] roxy\_pipeop\_torch\_construction

*   \[ ] roxy\_param\_module\_generator

*   \[ ] roxy\_pipeop\_torch\_license

*   \[ ] roxy\_param\_innum

*   \[ ] roxy\_param\_packages

*   \[ ] roxy\_pipeop\_torch\_format

*   \[ ] roxy\_param\_id

*   \[ ] roxy\_param\_param\_vals

*   \[ ] roxy\_pipeop\_torch\_channels\_default

*   \[ ] roxy\_param\_param\_set

*   \[ ] roxy\_pipeop\_torch\_param\_id

*   \[ ] roxy\_pipeop\_torch\_state\_default

*   \[ ] roxy\_construction

*   \[ ] roxy\_pipeop\_torch\_methods\_default

# Other

*   early\_stopping -> test renaming

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
    \--> properly tagging parameters and properly define learner methods.

*   general method for freezing and unfreezing parameters.

**Mit Martin**

*   It must be documented how the output names are generated, when outputs of non-terminal nodes are used
    in "output\_map" ("output\_<id>\_output.<channel>")
