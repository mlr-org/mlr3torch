#' @title Builds a mlr3 Torch Learner from its input
#' @description Use TorchOpModeLClassif or TorchOpModelRegr for the respective task type.
#' During `$train()` this TorchOp first builds the model (network, optimizer,
#' loss, ...) and afterwards trains the network.
#' It's parameterset is identical to LearnerClassifTorch and LearnerRegrTorch respectively.
#' @details TorchOpModelClassif and TorchOpModelRegr inherit from this R6 Class.
#' @name mlr_torchops.model
#' @export
TorchOpModel = R6Class("TorchOpModel",
  inherit = TorchOp,
  public = list(
    #' @description Creates an object of class [TorchOpModel]\cr
    #' @param id (`character(1)`)\cr
    #'   The id of the object.
    #' @param param_vals (`named list()`)\cr
    #'   A list containing the initial parameter values.
    #' @param .task_type (`character(1)`)\cr
    #'   The task type, see mlr_reflections$task_types.
    #' @param .optimizer (`character(1)`)\cr
    #'   The optimizer, see torch_reflections$optimizer.
    #' @param .loss (`character(1)`)\cr
    #'   The loss function, see torch_reflections$loss.
    initialize = function(id, param_vals, .task_type, .optimizer, .loss) {
      assert_true(.optimizer %in% torch_reflections$optimizer)
      private$.optimizer = .optimizer
      private$.loss = .loss
      private$.task_type = assert_choice(.task_type, c("classif", "regr"))


      param_set = make_paramset(.task_type, .optimizer, .loss)
      param_set$add(
        ParamUty$new("architecture", tags = "train", custom_check = check_architecture)
      )

      input = data.table(name = "input", train = "ModelArgs", predict = "Task")
      output = data.table(name = "output", train = "NULL", predict = "Prediction")

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        output = output
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      input = inputs$input
      task = input$task

      # TODO: maybe the learner and the TorchOp should actually share the param-set?
      pars = self$param_set$get_values(tags = "train")
      if (!is.null(pars$architecture)) {
        stopf("Parameter 'architecture' was set, but is overwritten by ModelArgs")
      }

      class = switch(private$.task_type,
        regr = LearnerRegrTorch,
        classif = LearnerClassifTorch
      )
      learner = class$new(
        id = self$id,
        .optimizer = private$.optimizer,
        .loss = private$.loss,
        .feature_types = unique(task$feature_types$type)
      )

      learner$param_set$values = insert_named(pars, list(architecture = input$architecture))
      learner$properties = if (private$.task_type == "regr") {
        c("weights", "hotstart_forward")
      } else if (private$.task_type == "classif") {
        if (length(task$class_names) == 2L) { # 1L is set as multiclass
          c("twoclass", "weights", "hotstart_forward")
        } else {
          c("multiclass", "weights", "hotstart_forward")
        }
      }

      private$.learner = learner
      on.exit({
        private$.learner$state = NULL
      })
      self$state = private$.learner$train(input$task)$state

      list(NULL)
    },
    .predict = function(inputs) {
      # This is copied from mlr3pipelines (PipeOpLearner)
      on.exit({
        private$.learner$state = NULL
      })
      task = inputs[[1]]
      private$.learner$state = self$state
      list(private$.learner$predict(task))
    },
    .task_type = NULL,
    .learner = NULL,
    .optimizer = NULL,
    .loss = NULL
  )
)

#' @title Classification Model
#' @description
#' Classification model.
#' @export
TorchOpModelClassif = R6Class(
  inherit = TorchOpModel,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param .optimizer (`character(1)`)\cr
    #'   The optimizer.
    #' @param .loss (`character(1)`)\cr
    #'   The loss function.
    initialize = function(id = "model", param_vals = list(), .optimizer, .loss) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .task_type = "classif",
        .optimizer = .optimizer,
        .loss = .loss
      )
    }
  )
)

#' @title Regression Model
#' @description
#' Regression model.
#' @export
TorchOpModelRegr = R6Class(
  inherit = TorchOpModel,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param .optimizer (`character(1)`)\cr
    #'   The optimizer.
    #' @param .loss (`character(1)`)\cr
    #'   The loss function.
    initialize = function(id = "model", param_vals = list(), .optimizer, .loss) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        .task_type = "regr",
        .optimizer = .optimizer,
        .loss = .loss
      )
    }
  )
)

#' @include mlr_torchops.R
mlr_torchops$add("model.regr", value = TorchOpModelRegr)

#' @include mlr_torchops.R
mlr_torchops$add("model.classif", value = TorchOpModelClassif)
