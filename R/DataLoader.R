#' @export
palmer_penguins = torch::dataset(
  initialize = function() {
    self$data = self$prepare_penguin_data()
  },
  .getitem = function(index) {
    x = self$data[index, 2:-1]
    y = self$data[index, 1]$to(torch_long())
    list(x, y)
  },
  .length = function() {
    self$data$size()[[1]]
  },
  prepare_penguin_data = function() {

    input = na.omit(penguins)
    # conveniently, the categorical data are already factors
    input$species = as.numeric(input$species)
    input$island = as.numeric(input$island)
    input$sex = as.numeric(input$sex)

    input = as.matrix(input)
    torch_tensor(input)
  }
)

DataLoader = R6Class("DataLoader",
  inherit = mlr3::DataBackend,
  public = list()

)
