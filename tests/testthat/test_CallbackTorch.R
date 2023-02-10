test_that("CallbackTorch works", {
  tcb = callback_torch("custom",
    public = list(
      initialize = function(msg) {
        self$msg = msg
      }
    ),
    on_epoch_begin = function(ctx) {
      print(self$msg)
    }
  )

})


test_that("Can retrieve predefined callback", {
  t_clbk("checkpoint")

})

test_that("All callbacks are in dictionary")


test_that("Callback dictionary can be converted")

test_that("Callbacks are being executed in the right order")


test_that("Callbacks are executed at the ...", {

})
