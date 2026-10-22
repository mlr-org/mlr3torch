# Cost of the forward pass of an nn_graph, as created from PipeOpTorch operators (e.g. via nn()), compared to
# running its Graph with $train() (as nn_graph did before) and to the same network written as a plain nn_module.
# Also breaks a training step down into forward, backward and optimizer step. CPU with one thread.
devtools::load_all(".")
torch_set_num_threads(1)
torch_manual_seed(1)

ms = function(f, n = 100) {
  for (i in 1:10) f()
  t0 = Sys.time()
  for (i in 1:n) f()
  1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")) / n
}
report = function(what, value) cat(sprintf("%-60s %7.2f ms\n", what, value))

d_in = 13L
task = as_task_regr(data.table(matrix(rnorm(100 * d_in), 100, dimnames = list(NULL, paste0("x", seq_len(d_in)))),
  y = rnorm(100)), target = "y")
block = nn("multihead_attention", num_heads = 1) %>>% nn("layer_norm", dims = 1) %>>%
  nn("linear", out_features = 32) %>>% nn("relu") %>>% nn("dropout", p = 0.3)
md = (po("torch_ingress_num") %>>% nn("tokenizer_num", d_token = 8) %>>% nn("block", block, n_blocks = 1) %>>%
  nn("flatten") %>>% nn("head"))$train(task)[[1L]]
net = model_descriptor_to_module(md)

# the same architecture as a plain nn_module
plain_net = nn_module(
  initialize = function(d_in) {
    self$tok = nn_tokenizer_num(n_features = d_in, d_token = 8, bias = TRUE, initialization = "uniform")
    self$mha = nn_multihead_attention(embed_dim = 8, num_heads = 1, batch_first = TRUE)
    self$ln = nn_layer_norm(8)
    self$lin = nn_linear(8, 32)
    self$drop = nn_dropout(0.3)
    self$head = nn_linear(d_in * 32, 1)
  },
  forward = function(x) {
    h = self$tok(x)
    h = self$mha(h, h, h, need_weights = FALSE)[[1L]]
    self$head(torch_flatten(self$drop(nnf_relu(self$lin(self$ln(h)))), start_dim = 2))
  }
)
plain = plain_net(d_in)

x = torch_randn(256, d_in)
y = torch_randn(256, 1)
step = function(model, opt) {
  opt$zero_grad()
  nnf_mse_loss(model(x), y)$backward()
  opt$step()
}

report("nn_graph, forward via Graph$train()",
  ms(function() with_no_grad(net$graph$train(list(x), single_input = FALSE))))
report("nn_graph, forward", ms(function() with_no_grad(net(x))))
report("plain nn_module, forward", ms(function() with_no_grad(plain(x))))
report("nn_graph, forward + backward", ms(function() {
  net$zero_grad()
  nnf_mse_loss(net(x), y)$backward()
}))
report("plain nn_module, forward + backward", ms(function() {
  plain$zero_grad()
  nnf_mse_loss(plain(x), y)$backward()
}))
for (optimizer in c("adamw", "ignite_adamw")) {
  opt = switch(optimizer, adamw = optim_adamw(net$parameters), ignite_adamw = optim_ignite_adamw(net$parameters))
  nnf_mse_loss(net(x), y)$backward()
  report(paste("optimizer step,", optimizer), ms(function() opt$step()))
}
report("nn_graph, training step with ignite_adamw", ms(function() step(net, optim_ignite_adamw(net$parameters))))
h = torch_randn(256, d_in, 8)
mha = nn_multihead_attention(embed_dim = 8, num_heads = 1, batch_first = TRUE)
report("nn_multihead_attention alone, forward", ms(function() with_no_grad(mha(h, h, h))))

# Results, batch size 256 (before mlr3torch ae5a990c, nn_graph ran Graph$train() in every forward pass):
# nn_graph, forward via Graph$train()                          121.95 ms (what nn_graph, forward took before)
# nn_graph, forward                                             13.58 ms
# plain nn_module, forward                                      12.19 ms
# nn_graph, forward + backward                                  25.30 ms
# plain nn_module, forward + backward                           22.86 ms
# optimizer step, adamw                                         40.15 ms (torch's R implementation)
# optimizer step, ignite_adamw                                   1.10 ms (used by t_opt("adamw"))
# nn_graph, training step with ignite_adamw                     26.55 ms
# nn_multihead_attention alone, forward                          6.80 ms (implemented in R in torch)
