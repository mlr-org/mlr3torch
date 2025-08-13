# printer

    Code
      lrn("classif.mlp", callbacks = list(t_clbk("history"), t_clbk("progress")))
    Output
      
      -- <LearnerTorchMLP> (classif.mlp): Multi Layer Perceptron ---------------------
      * Model: -
      * Parameters: device=auto, num_threads=1, num_interop_threads=1, seed=random,
      eval_freq=1, measures_train=<list>, measures_valid=<list>, patience=0,
      min_delta=0, shuffle=TRUE, tensor_dataset=FALSE, jit_trace=FALSE,
      neurons=integer(0), p=0.5, activation=<nn_relu>, activation_args=<list>
      * Validate: NULL
      * Packages: mlr3, mlr3torch, torch, and progress
      * Predict Types: [response] and prob
      * Feature Types: integer, numeric, and lazy_tensor
      * Encapsulation: none (fallback: -)
      * Properties: internal_tuning, marshal, multiclass, twoclass, and validation
      * Other settings: use_weights = 'error'
      * Optimizer: adam
      * Loss: cross_entropy
      * Callbacks: history,progress

