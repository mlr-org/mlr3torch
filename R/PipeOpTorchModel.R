#' @title Builds a mlr3 Torch Learner from its input
#' @description Use TorchOpModeLClassif or TorchOpModelRegr for the respective task type.
#' During `$train()` this TorchOp first builds the model (network, optimizer,
#' loss, ...) and afterwards trains the network.
#' It's parameterset is identical to LearnerClassifTorch and LearnerRegrTorch respectively.
#' @details TorchOpModelClassif and TorchOpModelRegr inherit from this R6 Class.
#' @name mlr_torchops.model
#' @export
PipeOpTorchModel = R6Class("PipeOpTorchModel",
  inherit = PipeOp,
  public = list(
    #' @description Creates an object of class [TorchOpModel]\cr
    #' @param id (`character(1)`)\cr
    #'   The id of the object.
    #' @param param_vals (`named list()`)\cr
    #'   A list containing the initial parameter values.
    #' @param task_type (`character(1)`)\cr
    #'   The task type, see mlr_reflections$task_types.
    #' @param optimizer (`character(1)`)\cr
    #'   The optimizer, see torch_reflections$optimizer.
    #' @param loss (`character(1)`)\cr
    #'   The loss function, see torch_reflections$loss.
    initialize = function(id, param_vals, task_type) {
      private$.task_type = assert_choice(task_type, c("classif", "regr"))

      param_set = paramset_torchlearner()

      input = data.table(name = "input", train = "ModelDescriptor", predict = mlr_reflections$task_types["classif", task])
      output = data.table(name = "output", train = "NULL", predict = mlr_reflections$task_types["classif", prediction])

      super$initialize(
        id = id,
        param_set = param_set,
        param_vals = param_vals,
        input = input,
        output = output
      )
    }
  ),
  private = list(
    .train = function(inputs) {
      md = inputs[[1]]
      param_vals = self$param_set$get_values()

      class = switch(private$.task_type,
        regr = LearnerRegrTorchModel,
        classif = LearnerClassifTorchModel
      )

      network = model_descriptor_to_module(md, list(md$.pointer))
      network$reset_parameters()

      learner = class$new(
        network = network,
        ingress_tokens = md$ingress,
        optimizer = md$optimizer,
        loss = md$loss,
        packages = md$graph$packages
      )

      learner$param_set$values = insert_named(learner$param_set$values, param_vals)

      learner$train(md$task)

      self$state = learner

      list(NULL)
    },
    .predict = function(inputs) {
      # This is copied from mlr3pipelines (PipeOpLearner)
      task = inputs[[1]]
      list(self$state$predict(task))
    },
    .task_type = NULL
  )
)

#' @title Classification Model
#' @description
#' Classification model.
#' @export
PipeOpTorchModelClassif = R6Class("PipeOpTorchModelClassif",
  inherit = PipeOpTorchModel,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param .optimizer (`character(1)`)\cr
    #'   The optimizer.
    #' @param loss (`character(1)`)\cr
    #'   The loss function.
    initialize = function(id = "torch_model.classif", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        task_type = "classif"
      )
    }
  )
)

#' @title Regression Model
#' @description
#' Regression model.
#' @export
PipeOpTorchModelRegr = R6Class("PipeOpTorchModelRegr",
  inherit = PipeOpTorchModel,
  public = list(
    #' @description Initializes an instance of this [R6][R6::R6Class] class.
    #' @param id (`character(1)`)\cr
    #'   The id for of the object.
    #' @param param_vals (named `list()`)\cr
    #'   The initial parameters for the object.
    #' @param optimizer (`character(1)`)\cr
    #'   The optimizer.
    #' @param loss (`character(1)`)\cr
    #'   The loss function.
    initialize = function(id = "torch_model.regr", param_vals = list()) {
      super$initialize(
        id = id,
        param_vals = param_vals,
        task_type = "regr"
      )
    }
  )
)

#' @include zzz.R
register_po("torch_model.regr", PipeOpTorchModelRegr)
register_po("torch_model.classif", PipeOpTorchModelClassif)
