% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PipeOpTorchLoss.R
\name{mlr_pipeops_torch_loss}
\alias{mlr_pipeops_torch_loss}
\alias{PipeOpTorchLoss}
\title{Loss Configuration}
\description{
Configures the loss of a deep learning model.
}
\section{Input and Output Channels}{

One input channel called \code{"input"} and one output channel called \code{"output"}.
For an explanation see \code{\link{PipeOpTorch}}.
}

\section{State}{

The state is the value calculated by the public method \code{shapes_out()}.
}

\section{Parameters}{

The parameters are defined dynamically from the loss set during construction.
}

\section{Internals}{

During training the loss is cloned and added to the \code{\link{ModelDescriptor}}.
}

\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
po_loss = po("torch_loss", loss = t_loss("cross_entropy"))
po_loss$param_set
mdin = po("torch_ingress_num")$train(list(tsk("iris")))
mdin[[1L]]$loss
mdout = po_loss$train(mdin)[[1L]]
mdout$loss
\dontshow{\}) # examplesIf}
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
\code{\link{mlr_pipeops_torch_ingress}},
\code{\link{mlr_pipeops_torch_ingress_categ}},
\code{\link{mlr_pipeops_torch_ingress_ltnsr}},
\code{\link{mlr_pipeops_torch_ingress_num}},
\code{\link{mlr_pipeops_torch_model}},
\code{\link{mlr_pipeops_torch_model_classif}},
\code{\link{mlr_pipeops_torch_model_regr}}

Other Model Configuration: 
\code{\link{ModelDescriptor}()},
\code{\link{mlr_pipeops_torch_callbacks}},
\code{\link{mlr_pipeops_torch_optimizer}},
\code{\link{model_descriptor_union}()}
}
\concept{Model Configuration}
\concept{PipeOps}
\section{Super class}{
\code{\link[mlr3pipelines:PipeOp]{mlr3pipelines::PipeOp}} -> \code{PipeOpTorchLoss}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-PipeOpTorchLoss-new}{\code{PipeOpTorchLoss$new()}}
\item \href{#method-PipeOpTorchLoss-clone}{\code{PipeOpTorchLoss$clone()}}
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
\if{html}{\out{<a id="method-PipeOpTorchLoss-new"></a>}}
\if{latex}{\out{\hypertarget{method-PipeOpTorchLoss-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PipeOpTorchLoss$new(loss, id = "torch_loss", param_vals = list())}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{loss}}{(\code{\link{TorchLoss}} or \code{character(1)} or \code{nn_loss})\cr
The loss (or something convertible via \code{\link[=as_torch_loss]{as_torch_loss()}}).}

\item{\code{id}}{(\code{character(1)})\cr
Identifier of the resulting  object.}

\item{\code{param_vals}}{(\code{list()})\cr
List of hyperparameter settings, overwriting the hyperparameter settings that would
otherwise be set during construction.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-PipeOpTorchLoss-clone"></a>}}
\if{latex}{\out{\hypertarget{method-PipeOpTorchLoss-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PipeOpTorchLoss$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
