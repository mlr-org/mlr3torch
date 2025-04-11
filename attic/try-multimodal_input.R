devtools::load_all()

withr::local_options(mlr3torch.cache = TRUE)

task_melanoma = tsk("melanoma")
task_melanoma$filter(1:5)

task_melanoma$data()

# adam = t_opt("adam", lr = 0.02)
# xe = t_loss("cross_entropy")
gr = gunion(list(
  po("select", id = "select_num", selector = selector_type("numeric")) %>>%
    po("torch_ingress_num", id = "ingress.num") %>>%
    po("nn_linear", out_features = 3, id = "linear1"),
  po("select", id = "select_categ", selector = selector_type("factor")) %>>%
    po("torch_ingress_categ", id = "ingress.categ") %>>%
    po("nn_linear", out_features = 3, id = "linear2"),
  po("select", id = "select_image", selector = selector_name("image")) %>>%
    po("torch_ingress_ltnsr", id = "ingress.image") %>>%
    po("nn_linear", out_features = 3, id = "linear3")
)) %>>%
  po("nn_merge_cat")  %>>%
  po("nn_relu", id = "act1") %>>%
  po("nn_linear", out_features = 3, id = "linear4") %>>%
  po("nn_softmax", dim = 2, id = "act3")

md = gr$train(task_melanoma)[[1L]]

net = nn_graph(md$graph, shapes_in = list(torch_ingress_num.input = c(NA, 4L),
                                          torch_ingress_categ.input = c(NA, 1L),
                                          torch_ingress_ltnsr.input = c(NA, 1L)))

# TODO: fill in some example concrete inputs
# i.e. select the relevant columns from the above filtered tasks
net(feats_num, feats_categ, feats_ltnsr)

