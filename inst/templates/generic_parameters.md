*Generic Parameters*:

* `batch_size` :: `integer(1)`\cr
  The batch size for training and evaluating the network.
* `epochs` :: `integer(1)`\cr
  The number of training epochs.
* `callbacks` :: `list()`\cr
  A list of Callbacks.
* `device` :: `character(1)`\cr
  The default to be used for training and evaluating the network.
* `measures` :: `character(1)`\cr
  A list of measures used for model validation.
  These are stored in the learner's history.
* `augmentation` :: `function`\cr
  A function that transforms the batches returned by the data-loader.
* `callbacks` :: `list()`\cr
  A list of Callbacks to customize the training process.
* `drop_last` :: `logical(1)`\cr
  Whether to drop the last batch in each epoch.
* `keep_last_prediction` :: `logical(1)`\cr
  Whether to keep the last prediction. If set to `TRUE` this can avoid superfluous predictions.
* `num_threads` :: `integer(1)`\cr
  The number of threads to be used during training. Only relevant if the training is on the
  CPU.
* `shuffle` :: `logical(1)`\cr
  Whether to shuffle the training data. This can be useful when the data is sorted.
* `early_stopping_rounds` :: `integer(1)`\cr
  The patience parameter for the early stopping.

The parameters for the optimizer and the loss function are dynamically inferred during construction.
Consult the help pages of the {torch} package for their description.

