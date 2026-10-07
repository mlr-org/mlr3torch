nn_recurrent = nn_module(
  "nn_recurrent",
  initialize = function(type, input_size, hidden_size, num_layers = 1, bias = TRUE, dropout = 0,
    bidirectional = FALSE, nonlinearity = NULL, return_state = FALSE) {
    self$type = assert_choice(type, c("rnn", "lstm", "gru"))
    self$return_state = assert_flag(return_state)
    args = list(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers,
      bias = bias, batch_first = TRUE, dropout = dropout, bidirectional = bidirectional)
    # only `nn_rnn()` has a choice of activation; leaving it out keeps torch's own default
    if (self$type == "rnn" && !is.null(nonlinearity)) {
      args$nonlinearity = nonlinearity
    }
    generator = switch(self$type, rnn = torch::nn_rnn, lstm = torch::nn_lstm, gru = torch::nn_gru)
    self$rnn = do.call(generator, args)
  },
  forward = function(input) {
    out = self$rnn(input)
    if (!self$return_state) {
      return(out[[1L]])
    }
    # The final state is `(layers * directions, batch, hidden)` even in the batch-first layout, so
    # it is transposed to put the batch dimension first, where every tensor in a network has it.
    state = out[[2L]]
    if (self$type == "lstm") {
      list(output = out[[1L]], h_n = state[[1L]]$transpose(1L, 2L), c_n = state[[2L]]$transpose(1L, 2L)) # nolint
    } else {
      list(output = out[[1L]], h_n = state$transpose(1L, 2L))
    }
  }
)

# The parameters that all three recurrent layers share. `nn_rnn()` adds `nonlinearity` to them.
paramset_recurrent = function() {
  ps(
    hidden_size = p_int(lower = 1L, tags = c("train", "required")),
    num_layers = p_int(lower = 1L, default = 1L, tags = "train"),
    bias = p_lgl(default = TRUE, tags = "train"),
    dropout = p_dbl(lower = 0, upper = 1, default = 0, tags = "train"),
    bidirectional = p_lgl(default = FALSE, tags = "train")
  )
}

#' @title Recurrent Layer
#'
#' @description
#' Base class for the recurrent layers [`PipeOpTorchRNN`], [`PipeOpTorchLSTM`] and
#' [`PipeOpTorchGRU`].
#'
#' @section Parameters: See the respective child class.
#'
#' @name mlr_pipeops_nn_recurrent
#' @template pipeop_torch_state_default
#' @section Input and Output Channels:
#' One input channel `"input"`, and the output channels described in the child classes.
#' For an explanation see [`PipeOpTorch`].
#'
#' @family PipeOps
#' @include PipeOpTorch.R
#' @export
PipeOpTorchRecurrent = R6Class("PipeOpTorchRecurrent",
  inherit = PipeOpTorch,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    #' @template param_param_set
    #' @param type (`character(1)`)\cr
    #'   Which recurrent layer to build, one of `"rnn"`, `"lstm"` or `"gru"`.
    initialize = function(id, type, param_set, param_vals = list()) {
      private$.type = assert_choice(type, c("rnn", "lstm", "gru"))
      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        module_generator = nn_recurrent,
        # `nn_lstm()` carries a cell state in addition to the hidden state, the other two do not
        outname = if (private$.type == "lstm") c("output", "h_n", "c_n") else c("output", "h_n"),
        tags = "abstract"
      )
      # the final states are only output channels when they are requested
      self$outputs = "output"
    }
  ),
  private = list(
    .type = NULL,
    # whether any of the final states is among `$outputs`, in which case all of them are computed
    .return_state = function() {
      !identical(self$output$name, "output")
    },
    # torch returns the final states together with the output sequence
    .module_outputs = function() {
      if (private$.return_state()) private$.output_all$name else "output"
    },
    .additional_phash_input = function() {
      c(super$.additional_phash_input(), list(private$.type))
    },
    .shapes_out = function(shapes_in, param_vals, task) {
      shape = shapes_in[[1L]]
      assert_ndim(shape, 3L, self$id)
      # the last dimension becomes `input_size`, which the module needs to size its weights, whereas
      # the sequence length is only needed at runtime and may stay unknown
      assert_known_dims(shape, 3L, "the last dimension (which becomes 'input_size')", self$id)
      hidden_size = param_vals[["hidden_size"]]
      directions = if (isTRUE(param_vals[["bidirectional"]])) 2L else 1L
      num_layers = param_vals[["num_layers"]] %??% 1L
      # a bidirectional layer concatenates the two directions along the feature dimension
      output = as.integer(c(shape[1:2], hidden_size * directions))
      # the state carries one vector per layer and direction, transposed to be batch-first
      state = as.integer(c(shape[[1L]], num_layers * directions, hidden_size))
      c(list(output), rep(list(state), nrow(private$.output_all) - 1L))
    },
    .shape_dependent_params = function(shapes_in, param_vals, task) {
      param_vals$input_size = shapes_in[[1L]][[3L]]
      param_vals$type = private$.type
      # the final states are only computed when they are among `$outputs`
      param_vals$return_state = private$.return_state()
      param_vals
    }
  )
)

#' @title Simple Recurrent Layer
#' @inherit torch::nn_rnn description
#' @section Tensor Layout:
#' Input and output are `(batch, sequence, feature)`, i.e. the `batch_first` layout of
#' [`torch::nn_rnn()`], which is fixed and not a hyperparameter.
#' `torch` defaults to `(sequence, batch, feature)`, but the first dimension of every shape has to
#' be the batch dimension here.
#'
#' The final hidden state is `(layers * directions, batch, hidden_size)` in `torch`, even in the
#' batch-first layout. It is transposed to `(batch, layers * directions, hidden_size)` here, so that
#' the batch dimension comes first for it as well.
#'
#' @section nn_module:
#' Calls [`torch::nn_rnn()`] when trained, where the parameter `input_size` is inferred as the last
#' dimension of the input tensor and `batch_first` is always `TRUE`, see section *Tensor Layout*.
#'
#' @section Parameters:
#' * `hidden_size` :: `integer(1)`\cr
#'   The number of features of the hidden state.
#' * `num_layers` :: `integer(1)`\cr
#'   The number of stacked recurrent layers. Default is `1`.
#' * `bias` :: `logical(1)`\cr
#'   Whether to use bias weights. Default is `TRUE`.
#' * `dropout` :: `numeric(1)`\cr
#'   Dropout probability on the output of each layer but the last. Only has an effect when
#'   `num_layers` is greater than `1`. Default is `0`.
#' * `bidirectional` :: `logical(1)`\cr
#'   Whether to run the sequence in both directions and concatenate the results, which doubles the
#'   size of the feature dimension of the output. Default is `FALSE`.
#' * `nonlinearity` :: `character(1)`\cr
#'   The activation function, either `"tanh"` or `"relu"`. Default is `"tanh"`.
#'
#' Note that `input_size` is *not* a parameter, as it is inferred from the shape of the input
#' tensor, and that `batch_first` is *not* a parameter either, as it is fixed to `TRUE`, see section
#' *Tensor Layout*.
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"`, the sequence to run over.
#'
#' The module has two outputs: `"output"`, the output sequence of shape
#' `(batch, sequence, hidden_size * directions)`, and `"h_n"`, the final hidden state of shape
#' `(batch, num_layers * directions, hidden_size)`.
#' Only `"output"` is an output channel by default. Set `$outputs` to also (or only) get the final
#' state, e.g. `outputs = c("output", "h_n")`.
#'
#' For an explanation see [`PipeOpTorch`].
#'
#' @templateVar id nn_rnn
#' @templateVar param_vals hidden_size = 10
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchRNN = R6Class("PipeOpTorchRNN",
  inherit = PipeOpTorchRecurrent,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_rnn", param_vals = list()) {
      param_set = c(paramset_recurrent(), ps(
        nonlinearity = p_fct(default = "tanh", levels = c("tanh", "relu"), tags = "train")
      ))
      super$initialize(id = id, type = "rnn", param_set = param_set,
        param_vals = param_vals)
    }
  )
)

#' @title Long Short-Term Memory
#' @inherit torch::nn_lstm description
#' @inheritSection mlr_pipeops_nn_rnn Tensor Layout
#' @section nn_module:
#' Calls [`torch::nn_lstm()`] when trained, where the parameter `input_size` is inferred as the last
#' dimension of the input tensor and `batch_first` is always `TRUE`, see section *Tensor Layout*.
#'
#' @section Parameters:
#' The parameters are those of [`nn("rnn")`][mlr_pipeops_nn_rnn] without `nonlinearity`, whose
#' activation functions an LSTM cell fixes.
#'
#' @section Input and Output Channels:
#' There is one input channel `"input"`, the sequence to run over.
#'
#' The module has three outputs: `"output"`, the output sequence of shape
#' `(batch, sequence, hidden_size * directions)`, and `"h_n"` and `"c_n"`, the final hidden state and
#' the final cell state, both of shape `(batch, num_layers * directions, hidden_size)`.
#' Only `"output"` is an output channel by default. Set `$outputs` to also (or only) get the final
#' states, e.g. `outputs = c("output", "h_n", "c_n")`.
#'
#' For an explanation see [`PipeOpTorch`].
#'
#' @templateVar id nn_lstm
#' @templateVar param_vals hidden_size = 10
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchLSTM = R6Class("PipeOpTorchLSTM",
  inherit = PipeOpTorchRecurrent,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_lstm", param_vals = list()) {
      super$initialize(id = id, type = "lstm", param_set = paramset_recurrent(),
        param_vals = param_vals)
    }
  )
)

#' @title Gated Recurrent Unit
#' @inherit torch::nn_gru description
#' @inheritSection mlr_pipeops_nn_rnn Tensor Layout
#' @section nn_module:
#' Calls [`torch::nn_gru()`] when trained, where the parameter `input_size` is inferred as the last
#' dimension of the input tensor and `batch_first` is always `TRUE`, see section *Tensor Layout*.
#'
#' @inheritSection mlr_pipeops_nn_lstm Parameters
#' @inheritSection mlr_pipeops_nn_rnn Input and Output Channels
#'
#' @templateVar id nn_gru
#' @templateVar param_vals hidden_size = 10
#' @template pipeop_torch
#' @template pipeop_torch_example
#'
#' @export
PipeOpTorchGRU = R6Class("PipeOpTorchGRU",
  inherit = PipeOpTorchRecurrent,
  public = list(
    #' @description Creates a new instance of this [R6][R6::R6Class] class.
    #' @template params_pipelines
    initialize = function(id = "nn_gru", param_vals = list()) {
      super$initialize(id = id, type = "gru", param_set = paramset_recurrent(),
        param_vals = param_vals)
    }
  )
)

#' @include aaa.R
register_po("nn_rnn", PipeOpTorchRNN)
register_po("nn_lstm", PipeOpTorchLSTM)
register_po("nn_gru", PipeOpTorchGRU)
