% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PipeOpTorchIngress.R
\name{mlr_pipeops_torch_ingress}
\alias{mlr_pipeops_torch_ingress}
\alias{PipeOpTorchIngress}
\title{Entrypoint to Torch Network}
\description{
Use this as entry-point to mlr3torch-networks.
Unless you are an advanced user, you should not need to use this directly but \code{\link{PipeOpTorchIngressNumeric}},
\code{\link{PipeOpTorchIngressCategorical}} or \code{\link{PipeOpTorchIngressLazyTensor}}.
}
\section{Parameters}{

Defined by the construction argument \code{param_set}.
}

\section{Input and Output Channels}{

One input channel called \code{"input"} and one output channel called \code{"output"}.
For an explanation see \code{\link{PipeOpTorch}}.
}

\section{State}{

The state is set to the input shape.
}

\section{Internals}{

Creates an object of class \code{\link{TorchIngressToken}} for the given task.
The purpuse of this is to store the information on how to construct the torch dataloader from the task for this
entry point of the network.
}

\seealso{
Other PipeOps: 
\code{\link{mlr_pipeops_nn_adaptive_avg_pool1d}},
\code{\link{mlr_pipeops_nn_adaptive_avg_pool2d}},
\code{\link{mlr_pipeops_nn_adaptive_avg_pool3d}},
\code{\link{mlr_pipeops_nn_avg_pool1d}},
\code{\link{mlr_pipeops_nn_avg_pool2d}},
\code{\link{mlr_pipeops_nn_avg_pool3d}},
\code{\link{mlr_pipeops_nn_batch_norm1d}},
\code{\link{mlr_pipeops_nn_batch_norm2d}},
\code{\link{mlr_pipeops_nn_batch_norm3d}},
\code{\link{mlr_pipeops_nn_block}},
\code{\link{mlr_pipeops_nn_celu}},
\code{\link{mlr_pipeops_nn_conv1d}},
\code{\link{mlr_pipeops_nn_conv2d}},
\code{\link{mlr_pipeops_nn_conv3d}},
\code{\link{mlr_pipeops_nn_conv_transpose1d}},
\code{\link{mlr_pipeops_nn_conv_transpose2d}},
\code{\link{mlr_pipeops_nn_conv_transpose3d}},
\code{\link{mlr_pipeops_nn_dropout}},
\code{\link{mlr_pipeops_nn_elu}},
\code{\link{mlr_pipeops_nn_flatten}},
\code{\link{mlr_pipeops_nn_ft_cls}},
\code{\link{mlr_pipeops_nn_ft_transformer_block}},
\code{\link{mlr_pipeops_nn_geglu}},
\code{\link{mlr_pipeops_nn_gelu}},
\code{\link{mlr_pipeops_nn_glu}},
\code{\link{mlr_pipeops_nn_hardshrink}},
\code{\link{mlr_pipeops_nn_hardsigmoid}},
\code{\link{mlr_pipeops_nn_hardtanh}},
\code{\link{mlr_pipeops_nn_head}},
\code{\link{mlr_pipeops_nn_identity}},
\code{\link{mlr_pipeops_nn_layer_norm}},
\code{\link{mlr_pipeops_nn_leaky_relu}},
\code{\link{mlr_pipeops_nn_linear}},
\code{\link{mlr_pipeops_nn_log_sigmoid}},
\code{\link{mlr_pipeops_nn_max_pool1d}},
\code{\link{mlr_pipeops_nn_max_pool2d}},
\code{\link{mlr_pipeops_nn_max_pool3d}},
\code{\link{mlr_pipeops_nn_merge}},
\code{\link{mlr_pipeops_nn_merge_cat}},
\code{\link{mlr_pipeops_nn_merge_prod}},
\code{\link{mlr_pipeops_nn_merge_sum}},
\code{\link{mlr_pipeops_nn_prelu}},
\code{\link{mlr_pipeops_nn_reglu}},
\code{\link{mlr_pipeops_nn_relu}},
\code{\link{mlr_pipeops_nn_relu6}},
\code{\link{mlr_pipeops_nn_reshape}},
\code{\link{mlr_pipeops_nn_rrelu}},
\code{\link{mlr_pipeops_nn_selu}},
\code{\link{mlr_pipeops_nn_sigmoid}},
\code{\link{mlr_pipeops_nn_softmax}},
\code{\link{mlr_pipeops_nn_softplus}},
\code{\link{mlr_pipeops_nn_softshrink}},
\code{\link{mlr_pipeops_nn_softsign}},
\code{\link{mlr_pipeops_nn_squeeze}},
\code{\link{mlr_pipeops_nn_tanh}},
\code{\link{mlr_pipeops_nn_tanhshrink}},
\code{\link{mlr_pipeops_nn_threshold}},
\code{\link{mlr_pipeops_nn_tokenizer_categ}},
\code{\link{mlr_pipeops_nn_tokenizer_num}},
\code{\link{mlr_pipeops_nn_unsqueeze}},
\code{\link{mlr_pipeops_torch_ingress_categ}},
\code{\link{mlr_pipeops_torch_ingress_ltnsr}},
\code{\link{mlr_pipeops_torch_ingress_num}},
\code{\link{mlr_pipeops_torch_loss}},
\code{\link{mlr_pipeops_torch_model}},
\code{\link{mlr_pipeops_torch_model_classif}},
\code{\link{mlr_pipeops_torch_model_regr}}

Other Graph Network: 
\code{\link{ModelDescriptor}()},
\code{\link{TorchIngressToken}()},
\code{\link{mlr_learners_torch_model}},
\code{\link{mlr_pipeops_module}},
\code{\link{mlr_pipeops_torch}},
\code{\link{mlr_pipeops_torch_ingress_categ}},
\code{\link{mlr_pipeops_torch_ingress_ltnsr}},
\code{\link{mlr_pipeops_torch_ingress_num}},
\code{\link{model_descriptor_to_learner}()},
\code{\link{model_descriptor_to_module}()},
\code{\link{model_descriptor_union}()},
\code{\link{nn_graph}()}
}
\concept{Graph Network}
\concept{PipeOps}
\section{Super class}{
\code{\link[mlr3pipelines:PipeOp]{mlr3pipelines::PipeOp}} -> \code{PipeOpTorchIngress}
}
\section{Active bindings}{
\if{html}{\out{<div class="r6-active-bindings">}}
\describe{
\item{\code{feature_types}}{(\code{character(1)})\cr
The features types that can be consumed by this \code{PipeOpTorchIngress}.}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-PipeOpTorchIngress-new}{\code{PipeOpTorchIngress$new()}}
\item \href{#method-PipeOpTorchIngress-clone}{\code{PipeOpTorchIngress$clone()}}
}
}
\if{html}{\out{
<details open><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="mlr3pipelines" data-topic="PipeOp" data-id="help"><a href='../../mlr3pipelines/html/PipeOp.html#method-PipeOp-help'><code>mlr3pipelines::PipeOp$help()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3pipelines" data-topic="PipeOp" data-id="predict"><a href='../../mlr3pipelines/html/PipeOp.html#method-PipeOp-predict'><code>mlr3pipelines::PipeOp$predict()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3pipelines" data-topic="PipeOp" data-id="print"><a href='../../mlr3pipelines/html/PipeOp.html#method-PipeOp-print'><code>mlr3pipelines::PipeOp$print()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3pipelines" data-topic="PipeOp" data-id="train"><a href='../../mlr3pipelines/html/PipeOp.html#method-PipeOp-train'><code>mlr3pipelines::PipeOp$train()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-PipeOpTorchIngress-new"></a>}}
\if{latex}{\out{\hypertarget{method-PipeOpTorchIngress-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PipeOpTorchIngress$new(
  id,
  param_set = ps(),
  param_vals = list(),
  packages = character(0),
  feature_types
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{id}}{(\code{character(1)})\cr
Identifier of the resulting  object.}

\item{\code{param_set}}{(\code{\link[paradox:ParamSet]{ParamSet}})\cr
The parameter set.}

\item{\code{param_vals}}{(\code{list()})\cr
List of hyperparameter settings, overwriting the hyperparameter settings that would
otherwise be set during construction.}

\item{\code{packages}}{(\code{character()})\cr
The R packages this object depends on.}

\item{\code{feature_types}}{(\code{character()})\cr
The feature types.
See \code{\link[mlr3:mlr_reflections]{mlr_reflections$task_feature_types}} for available values,
Additionally, \code{"lazy_tensor"} is supported.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-PipeOpTorchIngress-clone"></a>}}
\if{latex}{\out{\hypertarget{method-PipeOpTorchIngress-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PipeOpTorchIngress$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
