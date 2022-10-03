* save the generated PipeOpModule in the state
* don't throw an error when PipeOpTorchIngressNumeric gets something with factors.
  We only throw an error later when there are some features present in the task that are not covered by any ingress token.
  This is done in PipeOpTorchModule


  todo tomorro:
  * fix channels PipeOpTorch (PipeOpTorchLinear fails)
  * finish documentation schema of pipeoptorch
  * finish autotest for pipeoptorch
