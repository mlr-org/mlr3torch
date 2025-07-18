% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LearnerTorchFeatureless.R
\name{mlr_learners.torch_featureless}
\alias{mlr_learners.torch_featureless}
\alias{LearnerTorchFeatureless}
\title{Featureless Torch Learner}
\description{
Featureless torch learner.
Output is a constant weight that is learned during training.
For classification, this should (asymptoptically) result in a majority class prediction when using the standard cross-entropy loss.
For regression, this should result in the median for L1 loss and in the mean for L2 loss.
}
\section{Dictionary}{

This \link[mlr3:Learner]{Learner} can be instantiated using the sugar function \code{\link[mlr3:mlr_sugar]{lrn()}}:

\if{html}{\out{<div class="sourceCode">}}\preformatted{lrn("classif.torch_featureless", ...)
lrn("regr.torch_featureless", ...)
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
\item Feature Types: \dQuote{logical}, \dQuote{integer}, \dQuote{numeric}, \dQuote{character}, \dQuote{factor}, \dQuote{ordered}, \dQuote{POSIXct}, \dQuote{Date}, \dQuote{lazy_tensor}
\item Required Packages: \CRANpkg{mlr3}, \CRANpkg{mlr3torch}, \CRANpkg{torch}
}
}

\section{Parameters}{

Only those from \code{\link{LearnerTorch}}.
}

\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
# Define the Learner and set parameter values
learner = lrn("classif.torch_featureless")
learner$param_set$set_values(
  epochs = 1, batch_size = 16, device = "cpu"
  
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
\seealso{
Other Learner: 
\code{\link{mlr_learners.ft_transformer}},
\code{\link{mlr_learners.mlp}},
\code{\link{mlr_learners.module}},
\code{\link{mlr_learners.tab_resnet}},
\code{\link{mlr_learners_torch}},
\code{\link{mlr_learners_torch_image}},
\code{\link{mlr_learners_torch_model}}
}
\concept{Learner}
\section{Super classes}{
\code{\link[mlr3:Learner]{mlr3::Learner}} -> \code{\link[mlr3torch:LearnerTorch]{mlr3torch::LearnerTorch}} -> \code{LearnerTorchFeatureless}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-LearnerTorchFeatureless-new}{\code{LearnerTorchFeatureless$new()}}
\item \href{#method-LearnerTorchFeatureless-clone}{\code{LearnerTorchFeatureless$clone()}}
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
\if{html}{\out{<a id="method-LearnerTorchFeatureless-new"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchFeatureless-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchFeatureless$new(
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
\if{html}{\out{<a id="method-LearnerTorchFeatureless-clone"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchFeatureless-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchFeatureless$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
