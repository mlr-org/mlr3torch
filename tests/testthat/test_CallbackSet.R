test_that("Basic checks", {
  expect_class(CallbackSet, "R6ClassGenerator")
  instance = CallbackSet$new()
  expect_true(is.null(CallbackSet$inherit))
  expect_true(!inherits(instance, "Callback"))
})

test_that("callback_set is working", {
  expect_subset(mlr_reflections$torch$callback_stages, formalArgs(callback_set))
  expect_subset(formalArgs(callback_set), formalArgs(torch_callback))

  expect_error(callback_set("A"), regexp = "startsWith")
  tcb = callback_set("CallbackSetA")
  expect_class(tcb, "R6ClassGenerator")
  expect_warning(callback_set("CallbackSetA", public = list(on_edn = function() NULL)), regexp = "on_edn")

  e = new.env()
  e$aaaabbb = 1441
  CallbackSetB = callback_set("CallbackSetB",
    public = list(
      a = 1
    ),
    private = list(
      b = 2
    ),
    active = list(
      c = function() 3
    ),
    parent_env = e
  )
  expect_class(CallbackSetB, "R6ClassGenerator")

  expect_identical(parent.env(CallbackSetB$parent_env), e)
  cb = CallbackSetB$new()
  expect_class(cb, "CallbackSetB")
  expect_identical(cb$a, 1)
  expect_identical(get_private(cb)$b, 2)
  expect_identical(cb$c, 3)

  A = R6Class("A")
  expect_error(callback_set("CallbackSetA", inherit = A), regexp = "does not generate object")
  B = R6Class("B", inherit = CallbackSet)
  expect_error(callback_set("CallbackSetA", inherit = B), regexp = NA)


  CallbackSetC = callback_set("CallbackSetC",
    initialize = function(x) {
      self$x = x
    }
  )

  cb = CallbackSetC$new(1)
  expect_equal(cb$x, 1)

  CallbackSetD = callback_set("CallbackSetD",
    public = list(
      initialize = function(x) {
        self$x = x
      }
    )
  )
  cb = CallbackSetC$new(1)
  expect_equal(cb$x, 1)

  expect_error(
    callback_set("CallbackSetE", public = list(initialize = function() NULL), initialize = function() NULL),
    "initialize"
  )

  CallbackSetF = callback_set("CallbackSetF",
    private = list(deep_clone = function(name, value) value)
  )
  expect_true(CallbackSetF$cloneable)

  CallbackSetG = callback_set("CallbackSetG")
  expect_false(CallbackSetG$cloneable)

  CallbackSetH = callback_set("CallbackSetTestH", initialize = function(ctx) NULL)
  expect_error(TorchCallback$new(CallbackSetH), "is reserved for the ContextTorch")
})


test_that("phash works", {
  expect_equal(t_clbk("checkpoint", freq = 1)$phash, t_clbk("checkpoint", freq = 2)$phash)
  expect_false(t_clbk("history")$phash == t_clbk("progress")$phash)
  expect_false(t_clbk("history", id = "a")$phash == t_clbk("history", id = "b")$phash)
  expect_false(t_clbk("history", label = "a")$phash == t_clbk("history", label = "b")$phash)
})
