test_that("PipeOpModel works", {
  ds = tensor_dataset(x = torch_randn(160, 10), y = torch_randn(160, 1))
  dl = dataloader(ds, batch_size = 2, shuffle = FALSE)

  graph = PipeOpLinear$new() 

  coro::loop(for (batch in dl) {
  })

})
