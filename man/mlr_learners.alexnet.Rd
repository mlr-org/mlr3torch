% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LearnerTorchAlexNet.R
\name{mlr_learners.alexnet}
\alias{mlr_learners.alexnet}
\alias{LearnerTorchAlexNet}
\title{AlexNet Image Classifier}
\description{
Historic convolutional neural network for image classification.
}
\section{Dictionary}{

This \link{Learner} can be instantiated using the sugar function \code{\link[=lrn]{lrn()}}:

\if{html}{\out{<div class="sourceCode">}}\preformatted{lrn("classif.alexnet", ...)

}\if{html}{\out{</div>}}
}

\section{Meta Information}{

\itemize{
\item Supported task types: 'classif'
\item Predict Types:
\itemize{
\item classif: 'response', 'prob'
}
\item Feature Types: \dQuote{lazy_tensor}
\item Required Packages: \CRANpkg{mlr3}, \CRANpkg{mlr3torch}, \CRANpkg{torchvision}, \CRANpkg{magick}, \CRANpkg{torch}
}
}

\section{State}{

The state is a list with elements:
\itemize{
\item \code{network} :: The trained \link[torch:nn_module]{network}.
\item \code{optimizer} :: The \link[torch:optimizer]{optimizer} used to train the network.
\item \code{loss_fn} :: The \link[torch:nn_module]{loss} used to train the network.
\item \code{callbacks} :: The \link[=mlr_callback_set]{callbacks} used to train the network.
\item \code{seed} :: The actual seed that was / is used for training and prediction.
}
}

\section{Parameters}{

Parameters from \code{\link{LearnerTorchImage}} and
\itemize{
\item \code{pretrained} :: \code{logical(1)}\cr
Whether to use the pretrained model.
}
}

\references{
Krizhevsky, Alex, Sutskever, Ilya, Hinton, E. G (2017).
\dQuote{Imagenet classification with deep convolutional neural networks.}
\emph{Communications of the ACM}, \bold{60}(6), 84--90.
}
\seealso{
Other Learner: 
\code{\link{mlr_learners.mlp}},
\code{\link{mlr_learners.torch_featureless}},
\code{\link{mlr_learners_torch}},
\code{\link{mlr_learners_torch_image}},
\code{\link{mlr_learners_torch_model}}
}
\concept{Learner}
\section{Super classes}{
\code{\link[mlr3:Learner]{mlr3::Learner}} -> \code{\link[mlr3torch:LearnerTorch]{mlr3torch::LearnerTorch}} -> \code{\link[mlr3torch:LearnerTorchImage]{mlr3torch::LearnerTorchImage}} -> \code{LearnerTorchAlexNet}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-LearnerTorchAlexNet-new}{\code{LearnerTorchAlexNet$new()}}
\item \href{#method-LearnerTorchAlexNet-clone}{\code{LearnerTorchAlexNet$clone()}}
}
}
\if{html}{\out{
<details><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="base_learner"><a href='../../mlr3/html/Learner.html#method-Learner-base_learner'><code>mlr3::Learner$base_learner()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="help"><a href='../../mlr3/html/Learner.html#method-Learner-help'><code>mlr3::Learner$help()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="predict"><a href='../../mlr3/html/Learner.html#method-Learner-predict'><code>mlr3::Learner$predict()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="predict_newdata"><a href='../../mlr3/html/Learner.html#method-Learner-predict_newdata'><code>mlr3::Learner$predict_newdata()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="reset"><a href='../../mlr3/html/Learner.html#method-Learner-reset'><code>mlr3::Learner$reset()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="train"><a href='../../mlr3/html/Learner.html#method-Learner-train'><code>mlr3::Learner$train()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3torch" data-topic="LearnerTorch" data-id="format"><a href='../../mlr3torch/html/LearnerTorch.html#method-LearnerTorch-format'><code>mlr3torch::LearnerTorch$format()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3torch" data-topic="LearnerTorch" data-id="print"><a href='../../mlr3torch/html/LearnerTorch.html#method-LearnerTorch-print'><code>mlr3torch::LearnerTorch$print()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-LearnerTorchAlexNet-new"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchAlexNet-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchAlexNet$new(
  task_type,
  optimizer = NULL,
  loss = NULL,
  callbacks = list()
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{task_type}}{(\code{character(1)})\cr
The task type, either \verb{"classif}" or \code{"regr"}.}

\item{\code{optimizer}}{(\code{\link{TorchOptimizer}})\cr
The optimizer to use for training.
Per default, \emph{adam} is used.}

\item{\code{loss}}{(\code{\link{TorchLoss}})\cr
The loss used to train the network.
Per default, \emph{mse} is used for regression and \emph{cross_entropy} for classification.}

\item{\code{callbacks}}{(\code{list()} of \code{\link{TorchCallback}}s)\cr
The callbacks. Must have unique ids.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-LearnerTorchAlexNet-clone"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchAlexNet-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchAlexNet$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
