# Network Layers

Below is a list of neural network layers that are available in
`mlr3torch`.

| Key                                                                                                         | Label                                           |
|:------------------------------------------------------------------------------------------------------------|:------------------------------------------------|
| [nn_adaptive_avg_pool1d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool1d.html)   | 1D Adaptive Average Pooling                     |
| [nn_adaptive_avg_pool2d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool2d.html)   | 2D Adaptive Average Pooling                     |
| [nn_adaptive_avg_pool3d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_adaptive_avg_pool3d.html)   | 3D Adaptive Average Pooling                     |
| [nn_avg_pool1d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool1d.html)                     | 1D Average Pooling                              |
| [nn_avg_pool2d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool2d.html)                     | 2D Average Pooling                              |
| [nn_avg_pool3d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_avg_pool3d.html)                     | 3D Average Pooling                              |
| [nn_batch_norm1d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm1d.html)                 | 1D Batch Normalization                          |
| [nn_batch_norm2d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm2d.html)                 | 2D Batch Normalization                          |
| [nn_batch_norm3d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_batch_norm3d.html)                 | 3D Batch Normalization                          |
| [nn_block](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_block.html)                               | Block Repetition                                |
| [nn_celu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_celu.html)                                 | CELU Activation Function                        |
| [nn_conv1d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv1d.html)                             | 1D Convolution                                  |
| [nn_conv2d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv2d.html)                             | 2D Convolution                                  |
| [nn_conv3d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv3d.html)                             | 3D Convolution                                  |
| [nn_conv_transpose1d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose1d.html)         | Transpose 1D Convolution                        |
| [nn_conv_transpose2d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose2d.html)         | Transpose 2D Convolution                        |
| [nn_conv_transpose3d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_conv_transpose3d.html)         | Transpose 3D Convolution                        |
| [nn_dropout](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_dropout.html)                           | Dropout                                         |
| [nn_elu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_elu.html)                                   | ELU Activation Function                         |
| [nn_flatten](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_flatten.html)                           | Flattens a Tensor                               |
| [nn_fn](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_fn.html)                                     | Custom Function                                 |
| [nn_ft_cls](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_ft_cls.html)                             | CLS Token for FT-Transformer                    |
| [nn_ft_transformer_block](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_ft_transformer_block.html) | Single Transformer Block for the FT-Transformer |
| [nn_geglu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_geglu.html)                               | GeGLU Activation Function                       |
| [nn_gelu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_gelu.html)                                 | GELU Activation Function                        |
| [nn_glu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_glu.html)                                   | GLU Activation Function                         |
| [nn_hardshrink](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardshrink.html)                     | Hard Shrink Activation Function                 |
| [nn_hardsigmoid](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardsigmoid.html)                   | Hard Sigmoid Activation Function                |
| [nn_hardtanh](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_hardtanh.html)                         | Hard Tanh Activation Function                   |
| [nn_head](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_head.html)                                 | Output Head                                     |
| [nn_identity](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_identity.html)                         | Identity Layer                                  |
| [nn_layer_norm](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_layer_norm.html)                     | Layer Normalization                             |
| [nn_leaky_relu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_leaky_relu.html)                     | Leaky ReLU Activation Function                  |
| [nn_linear](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_linear.html)                             | Linear Layer                                    |
| [nn_log_sigmoid](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_log_sigmoid.html)                   | Log Sigmoid Activation Function                 |
| [nn_max_pool1d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool1d.html)                     | 1D Max Pooling                                  |
| [nn_max_pool2d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool2d.html)                     | 2D Max Pooling                                  |
| [nn_max_pool3d](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_max_pool3d.html)                     | 3D Max Pooling                                  |
| [nn_merge_cat](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_cat.html)                       | Merge by Concatenation                          |
| [nn_merge_prod](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_prod.html)                     | Merge by Product                                |
| [nn_merge_sum](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_merge_sum.html)                       | Merge by Summation                              |
| [nn_prelu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_prelu.html)                               | PReLU Activation Function                       |
| [nn_reglu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_reglu.html)                               | ReGLU Activation Function                       |
| [nn_relu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_relu.html)                                 | ReLU Activation Function                        |
| [nn_relu6](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_relu6.html)                               | ReLU6 Activation Function                       |
| [nn_reshape](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_reshape.html)                           | Reshape a Tensor                                |
| [nn_rrelu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_rrelu.html)                               | RReLU Activation Function                       |
| [nn_selu](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_selu.html)                                 | SELU Activation Function                        |
| [nn_sigmoid](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_sigmoid.html)                           | Sigmoid Activation Function                     |
| [nn_softmax](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softmax.html)                           | Softmax                                         |
| [nn_softplus](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softplus.html)                         | SoftPlus Activation Function                    |
| [nn_softshrink](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softshrink.html)                     | Soft Shrink Activation Function                 |
| [nn_softsign](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_softsign.html)                         | SoftSign Activation Function                    |
| [nn_squeeze](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_squeeze.html)                           | Squeeze a Tensor                                |
| [nn_tanh](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tanh.html)                                 | Tanh Activation Function                        |
| [nn_tanhshrink](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tanhshrink.html)                     | Tanh Shrink Activation Function                 |
| [nn_threshold](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_threshold.html)                       | Treshold Activation Function                    |
| [nn_tokenizer_categ](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tokenizer_categ.html)           | Categorical Tokenizer                           |
| [nn_tokenizer_num](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_tokenizer_num.html)               | Numeric Tokenizer                               |
| [nn_unsqueeze](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_nn_unsqueeze.html)                       | Unqueeze a Tensor                               |
