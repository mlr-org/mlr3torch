% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LearnerFTTransformer.R
\name{mlr_learners.ft_transformer}
\alias{mlr_learners.ft_transformer}
\alias{LearnerTorchFTTransformer}
\title{FT-Transformer}
\description{
Feature-Tokenizer Transformer for tabular data that can either work on \code{\link{lazy_tensor}} inputs
or on standard tabular features.

Some differences from the paper implementation: no attention compression, no option to have prenormalization in the first layer.

If training is unstable, consider a combination of standardizing features (e.g. using \code{po("scale")}), using an adaptive optimizer (e.g. Adam), reducing the learning rate,
and using a learning rate scheduler (see \code{\link{CallbackSetLRScheduler}} for options).
}
\section{Dictionary}{

This \link[mlr3:Learner]{Learner} can be instantiated using the sugar function \code{\link[mlr3:mlr_sugar]{lrn()}}:

\if{html}{\out{<div class="sourceCode">}}\preformatted{lrn("classif.ft_transformer", ...)
lrn("regr.ft_transformer", ...)
}\if{html}{\out{</div>}}
}

\section{Properties}{

\itemize{
\item Supported task types: 'classif', 'regr'
\item Predict Types:
\itemize{
\item classif: 'response', 'prob'
\item regr: 'response'
}
\item Feature Types: \dQuote{logical}, \dQuote{integer}, \dQuote{numeric}, \dQuote{factor}, \dQuote{ordered}, \dQuote{lazy_tensor}
\item Required Packages: \CRANpkg{mlr3}, \CRANpkg{mlr3torch}, \CRANpkg{torch}
}
}

\section{Parameters}{

Parameters from \code{\link{LearnerTorch}} and \code{\link{PipeOpTorchFTTransformerBlock}}, as well as:
\itemize{
\item \code{n_blocks} :: \code{integer(1)}\cr
The number of transformer blocks.
\item \code{d_token} :: \code{integer(1)}\cr
The dimension of the embedding.
\item \code{cardinalities} :: \code{integer(1)}\cr
The number of categories for each categorical feature. This only needs to be specified when working with \code{\link{lazy_tensor}} inputs.
\item \code{init_token} :: \code{character(1)}\cr
The initialization method for the embedding weights. Either "uniform" or "normal". "Uniform" by default.
\item \code{ingress_tokens} :: named \code{list()} or \code{NULL}\cr
A list of \code{TorchIngressToken}s. Only required when using lazy tensor features.
The names are either "num.input" or "categ.input", and the values are lazy tensor ingress tokens constructed by, e.g. \verb{ingress_ltnsr(<num_feat_name>)}.
}
}

\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
# Define the Learner and set parameter values
learner = lrn("classif.ft_transformer")
learner$param_set$set_values(
  epochs = 1, batch_size = 16, device = "cpu",
  n_blocks = 2, d_token = 32, ffn_d_hidden_multiplier = 4/3
)

# Define a Task
task = tsk("iris")

# Create train and test set
ids = partition(task)

# Train the learner on the training ids
learner$train(task, row_ids = ids$train)

# Make predictions for the test rows
predictions = learner$predict(task, row_ids = ids$test)

# Score the predictions
predictions$score()
\dontshow{\}) # examplesIf}
}
\references{
Gorishniy Y, Rubachev I, Khrulkov V, Babenko A (2021).
\dQuote{Revisiting Deep Learning  for Tabular Data.}
\emph{arXiv}, \bold{2106.11959}.
}
\seealso{
Other Learner: 
\code{\link{mlr_learners.mlp}},
\code{\link{mlr_learners.module}},
\code{\link{mlr_learners.tab_resnet}},
\code{\link{mlr_learners.torch_featureless}},
\code{\link{mlr_learners_torch}},
\code{\link{mlr_learners_torch_image}},
\code{\link{mlr_learners_torch_model}}
}
\concept{Learner}
\section{Super classes}{
\code{\link[mlr3:Learner]{mlr3::Learner}} -> \code{\link[mlr3torch:LearnerTorch]{mlr3torch::LearnerTorch}} -> \code{LearnerTorchFTTransformer}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-LearnerTorchFTTransformer-new}{\code{LearnerTorchFTTransformer$new()}}
\item \href{#method-LearnerTorchFTTransformer-clone}{\code{LearnerTorchFTTransformer$clone()}}
}
}
\if{html}{\out{
<details><summary>Inherited methods</summary>
<ul>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="base_learner"><a href='../../mlr3/html/Learner.html#method-Learner-base_learner'><code>mlr3::Learner$base_learner()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="configure"><a href='../../mlr3/html/Learner.html#method-Learner-configure'><code>mlr3::Learner$configure()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="encapsulate"><a href='../../mlr3/html/Learner.html#method-Learner-encapsulate'><code>mlr3::Learner$encapsulate()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="help"><a href='../../mlr3/html/Learner.html#method-Learner-help'><code>mlr3::Learner$help()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="predict"><a href='../../mlr3/html/Learner.html#method-Learner-predict'><code>mlr3::Learner$predict()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="predict_newdata"><a href='../../mlr3/html/Learner.html#method-Learner-predict_newdata'><code>mlr3::Learner$predict_newdata()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="reset"><a href='../../mlr3/html/Learner.html#method-Learner-reset'><code>mlr3::Learner$reset()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="selected_features"><a href='../../mlr3/html/Learner.html#method-Learner-selected_features'><code>mlr3::Learner$selected_features()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3" data-topic="Learner" data-id="train"><a href='../../mlr3/html/Learner.html#method-Learner-train'><code>mlr3::Learner$train()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3torch" data-topic="LearnerTorch" data-id="dataset"><a href='../../mlr3torch/html/LearnerTorch.html#method-LearnerTorch-dataset'><code>mlr3torch::LearnerTorch$dataset()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3torch" data-topic="LearnerTorch" data-id="format"><a href='../../mlr3torch/html/LearnerTorch.html#method-LearnerTorch-format'><code>mlr3torch::LearnerTorch$format()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3torch" data-topic="LearnerTorch" data-id="marshal"><a href='../../mlr3torch/html/LearnerTorch.html#method-LearnerTorch-marshal'><code>mlr3torch::LearnerTorch$marshal()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3torch" data-topic="LearnerTorch" data-id="print"><a href='../../mlr3torch/html/LearnerTorch.html#method-LearnerTorch-print'><code>mlr3torch::LearnerTorch$print()</code></a></span></li>
<li><span class="pkg-link" data-pkg="mlr3torch" data-topic="LearnerTorch" data-id="unmarshal"><a href='../../mlr3torch/html/LearnerTorch.html#method-LearnerTorch-unmarshal'><code>mlr3torch::LearnerTorch$unmarshal()</code></a></span></li>
</ul>
</details>
}}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-LearnerTorchFTTransformer-new"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchFTTransformer-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchFTTransformer$new(
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
\if{html}{\out{<a id="method-LearnerTorchFTTransformer-clone"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchFTTransformer-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchFTTransformer$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
