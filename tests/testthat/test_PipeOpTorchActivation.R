# LogSigmoid

test_that("PipeOpTorchLogSigmoid autotest", {
  po_test = po("nn_log_sigmoid")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_log_sigmoid", task)
})

test_that("PipeOpTorchLogSigmoid paramtest", {
  res = expect_paramset(po("nn_log_sigmoid"), nn_log_sigmoid)
  expect_paramtest(res)
})

# Sigmoid

test_that("PipeOpTorchSigmoid autotest", {
  po_test = po("nn_sigmoid")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_sigmoid", task)
})

test_that("PipeOpTorchSigmoid paramtest", {
  res = expect_paramset(po("nn_sigmoid"), nn_sigmoid)
  expect_paramtest(res)
})

# GELU

test_that("PipeOpTorchGELU autotest", {
  po_test = po("nn_gelu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_gelu", task)
})

test_that("PipeOpTorchGELU paramtest", {
  res = expect_paramset(po("nn_gelu"), nn_gelu)
  expect_paramtest(res)
})

# PipeOpTorchReLU

test_that("PipeOpTorchReLU autotest", {
  po_test = po("nn_relu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_relu", task)
})

test_that("PipeOpTorchReLU paramtest", {
  res = expect_paramset(po("nn_relu"), nn_relu)
  expect_paramtest(res)
})


# PipeOpTorchTanhShrink

test_that("PipeOpTorchTanhShrink autotest", {
  po_test = po("nn_tanhshrink")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_tanhshrink", task)
})

test_that("PipeOpTorchTanhShrink paramtest", {
  res = expect_paramset(po("nn_tanhshrink"), nn_tanhshrink)
  expect_paramtest(res)
})

# PipeOpTorchGLU

test_that("PipeOpTorchGLU autotest", {
  po_test = po("nn_glu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")
  expect_pipeop_torch(graph, "nn_glu", task)
})

test_that("PipeOpTorchGLU paramtest", {
  res = expect_paramset(po("nn_glu"), nn_glu)
  expect_paramtest(res)
})

# PipeOpTorchCelu

test_that("PipeOpTorchCelu autotest", {
  po_test = po("nn_celu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_celu", task)
})

test_that("PipeOpTorchCelu paramtest", {
  res = expect_paramset(po("nn_celu"), nn_celu)
  expect_paramtest(res)
})

# PipeOpTorchThreshold

test_that("PipeOpTorchThreshold autotest", {
  po_test = po("nn_threshold", threshold = 2, value = 3)
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_threshold", task)
})

test_that("PipeOpTorchThreshold paramtest", {
  res = expect_paramset(po("nn_threshold"), nn_threshold)
  expect_paramtest(res)
})


# PipeOpTorchRReLU

test_that("PipeOpTorchRReLU autotest", {
  po_test = po("nn_rrelu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_rrelu", task)
})

test_that("PipeOpTorchRReLU paramtest", {
  res = expect_paramset(po("nn_rrelu"), nn_rrelu)
  expect_paramtest(res)
})


# PipeOpTorchHardSigmoid

test_that("PipeOpTorchHardSigmoid autotest", {
  po_test = po("nn_hardsigmoid")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_hardsigmoid", task)
})


test_that("PipeOpTorchHardSigmoid paramtest", {
  res = expect_paramset(po("nn_hardsigmoid"), nn_hardsigmoid)
  expect_paramtest(res)
})


# PipeOpTorchPReLU

test_that("PipeOpTorchPReLU autotest", {
  po_test = po("nn_prelu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_prelu", task)
})

test_that("PipeOpTorchPReLU paramtest", {
  res = expect_paramset(po("nn_prelu"), nn_prelu)
  expect_paramtest(res)
})


# PipeOpTorchTanh

test_that("PipeOpTorchTanh autotest", {
  po_test = po("nn_tanh")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_tanh", task)
})

test_that("PipeOpTorchTanh paramtest", {
  res = expect_paramset(po("nn_tanh"), nn_tanh)
  expect_paramtest(res)
})

# PipeOpTorchLeakyReLU

test_that("PipeOpTorchLeakyReLU autotest", {
  po_test = po("nn_leaky_relu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_leaky_relu", task)
})

test_that("PipeOpTorchLeakyReLU paramtest", {
  res = expect_paramset(po("nn_leaky_relu"), nn_leaky_relu)
  expect_paramtest(res)
})

# PipeOpTorchRelu6

test_that("PipeOpTorchRelu6 autotest", {
  po_test = po("nn_relu6")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_relu6", task)
})

test_that("PipeOpTorchRelu6 paramtest", {
  res = expect_paramset(po("nn_relu6"), nn_relu6)
  expect_paramtest(res)
})


# PipeOpTorchELU


test_that("PipeOpTorchELU autotest", {
  po_test = po("nn_elu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_elu", task)

})

test_that("PipeOpTorchELU paramtest", {
  res = expect_paramset(po("nn_elu"), nn_elu)
  expect_paramtest(res)
})

# PipeOpTorchtSoftShrink

test_that("PipeOpTorchtSoftShrink autotest", {
  po_test = po("nn_softshrink")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_softshrink", task)

})

test_that("PipeOpTorchtSoftShrink paramtest", {
  res = expect_paramset(po("nn_softshrink"), nn_softshrink)
  expect_paramtest(res)
})


# PipeOpTorchHardShrink

test_that("PipeOpTorchHardShrink autotest", {
  po_test = po("nn_hardshrink")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_hardshrink", task)

})

test_that("PipeOpTorchHardShrink paramtest", {
  res = expect_paramset(po("nn_hardshrink"), nn_hardshrink)
  expect_paramtest(res)
})


# PipeOpTorchSoftPlus

test_that("PipeOpTorchSoftPlus autotest", {
  po_test = po("nn_softplus")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_softplus", task)

})

test_that("PipeOpTorchSoftPlus paramtest", {
  res = expect_paramset(po("nn_softplus"), nn_softplus)
  expect_paramtest(res)
})

# PipeOpTorchSELU

test_that("PipeOpTorchSELU autotest", {
  po_test = po("nn_selu")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_selu", task)

})

test_that("PipeOpTorchSELU paramtest", {
  res = expect_paramset(po("nn_selu"), nn_selu)
  expect_paramtest(res)
})


# PipeOpTorchSoftmax

test_that("PipeOpTorchSoftmax autotest", {
  po_test = po("nn_softmax", dim = 2)
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_softmax", task)

})

test_that("PipeOpTorchSoftmax paramtest", {
  res = expect_paramset(po("nn_softmax"), nn_softmax)
  expect_paramtest(res)
})


# PipeOpTorchSoftSign

test_that("PipeOpTorchSoftSign autotest", {
  po_test = po("nn_softsign")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_softsign", task)

})

test_that("PipeOpTorchSoftSign paramtest", {
  res = expect_paramset(po("nn_softsign"), nn_softsign)
  expect_paramtest(res)
})

# PipeOpTorchHardTanh

test_that("PipeOpTorchHardTanh autotest", {
  po_test = po("nn_hardtanh")
  graph = po("torch_ingress_num") %>>% po_test
  task = tsk("iris")

  expect_pipeop_torch(graph, "nn_hardtanh", task)

})

test_that("PipeOpTorchHardTanh paramtest", {
  res = expect_paramset(po("nn_hardtanh"), nn_hardtanh)
  expect_paramtest(res)
})
