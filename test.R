library(torch)

y_hat = torch_randn(10, 1)
y = torch_rand(10)

nnf_binary_cross_entropy_with_logits(y_hat, y)

head(nnf_binary_cross_entropy)
