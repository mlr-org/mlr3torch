# Tabular Learners

The table below shows all tabular deep learning learners that are added
by `mlr3torch`: An overview of all learners can also be found on the
[mlr-org website](https://mlr-org.com/learners.html).

| key                       | label                     | task_type | feature_types                                                                             | packages |
|:--------------------------|:--------------------------|:----------|:------------------------------------------------------------------------------------------|:---------|
| classif.ft_transformer    | FT-Transformer            | classif   | logical , integer , numeric , factor , ordered , lazy_tensor                              |          |
| classif.mlp               | Multi Layer Perceptron    | classif   | integer , numeric , lazy_tensor                                                           |          |
| classif.module            | Custom Module             | classif   | logical , integer , numeric , character , factor , ordered , POSIXct , Date , lazy_tensor |          |
| classif.tab_resnet        | Tabular ResNet            | classif   | integer , numeric , lazy_tensor                                                           |          |
| classif.torch_featureless | Featureless Torch Learner | classif   | logical , integer , numeric , character , factor , ordered , POSIXct , Date , lazy_tensor |          |
| classif.torch_model       | Torch Model               | classif   | logical , integer , numeric , character , factor , ordered , POSIXct , Date , lazy_tensor |          |
| regr.ft_transformer       | FT-Transformer            | regr      | logical , integer , numeric , factor , ordered , lazy_tensor                              |          |
| regr.mlp                  | Multi Layer Perceptron    | regr      | integer , numeric , lazy_tensor                                                           |          |
| regr.module               | Custom Module             | regr      | logical , integer , numeric , character , factor , ordered , POSIXct , Date , lazy_tensor |          |
| regr.tab_resnet           | Tabular ResNet            | regr      | integer , numeric , lazy_tensor                                                           |          |
| regr.torch_featureless    | Featureless Torch Learner | regr      | logical , integer , numeric , character , factor , ordered , POSIXct , Date , lazy_tensor |          |
| regr.torch_model          | Torch Model               | regr      | logical , integer , numeric , character , factor , ordered , POSIXct , Date , lazy_tensor |          |
