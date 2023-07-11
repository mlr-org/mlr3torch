#' <% task_types_vec = strsplit(task_types, ", ")[[1]] %>
#' <% param_vals = if (exists("param_vals", inherits = FALSE)) param_vals else character(0) %>
#' <% id = paste0(task_types_vec[1], ".", name)%>
#' <% task_id = default_task_id(lrn(id)) %>
#'
#' @examples
#' # Define the Learner and set parameter values
#' <%= sprintf("learner = lrn(\"%s\")", id)%>
#' learner$param_set$set_values(
#' <%= paste0("  ", paste0(c(param_vals, "batch_size = 1", "epochs = 1"), collapse = ", "))%>
#' )
#'
#' # Define a Task
#' <%= sprintf("task = tsk(\"%s\")", task_id)%>
#'
#' # Create train and test set
#' <%= sprintf("ids = partition(task)")%>
#'
#' # Train the learner on the training ids
#' <%= sprintf("learner$train(task, row_ids = ids$train)")%>
#'
#' # Make predictions for the test rows
#' <%= sprintf("predictions = learner$predict(task, row_ids = ids$test)")%>
#'
#' # Score the predictions
#' predictions$score()
