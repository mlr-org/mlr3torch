% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LearnerTorchImage.R
\name{mlr_learners_torch_image}
\alias{mlr_learners_torch_image}
\alias{LearnerTorchImage}
\title{Image Learner}
\description{
Base Class for Image Learners.
The features are assumed to be a single \code{\link{lazy_tensor}} column in RGB format.
}
\section{Parameters}{

Parameters include those inherited from \code{\link{LearnerTorch}} and the \code{param_set} construction argument.
}

\seealso{
Other Learner: 
\code{\link{mlr_learners.ft_transformer}},
\code{\link{mlr_learners.mlp}},
\code{\link{mlr_learners.module}},
\code{\link{mlr_learners.tab_resnet}},
\code{\link{mlr_learners.torch_featureless}},
\code{\link{mlr_learners_torch}},
\code{\link{mlr_learners_torch_model}}
}
\concept{Learner}
\section{Super classes}{
\code{\link[mlr3:Learner]{mlr3::Learner}} -> \code{\link[mlr3torch:LearnerTorch]{mlr3torch::LearnerTorch}} -> \code{LearnerTorchImage}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-LearnerTorchImage-new}{\code{LearnerTorchImage$new()}}
\item \href{#method-LearnerTorchImage-clone}{\code{LearnerTorchImage$clone()}}
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
\if{html}{\out{<a id="method-LearnerTorchImage-new"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchImage-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchImage$new(
  id,
  task_type,
  param_set = ps(),
  label,
  optimizer = NULL,
  loss = NULL,
  callbacks = list(),
  packages,
  man,
  properties = NULL,
  predict_types = NULL,
  jittable = FALSE
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{id}}{(\code{character(1)})\cr
The id for of the new object.}

\item{\code{task_type}}{(\code{character(1)})\cr
The task type.}

\item{\code{param_set}}{(\code{\link[paradox:ParamSet]{ParamSet}})\cr
The parameter set.}

\item{\code{label}}{(\code{character(1)})\cr
Label for the new instance.}

\item{\code{optimizer}}{(\code{\link{TorchOptimizer}})\cr
The torch optimizer.}

\item{\code{loss}}{(\code{\link{TorchLoss}})\cr
The loss to use for training.}

\item{\code{callbacks}}{(\code{list()} of \code{\link{TorchCallback}}s)\cr
The callbacks used during training.
Must have unique ids.
They are executed in the order in which they are provided}

\item{\code{packages}}{(\code{character()})\cr
The R packages this object depends on.}

\item{\code{man}}{(\code{character(1)})\cr
String in the format \verb{[pkg]::[topic]} pointing to a manual page for this object.
The referenced help package can be opened via method \verb{$help()}.}

\item{\code{properties}}{(\code{character()})\cr
The properties of the object.
See \code{\link[mlr3:mlr_reflections]{mlr_reflections$learner_properties}} for available values.}

\item{\code{predict_types}}{(\code{character()})\cr
The predict types.
See \code{\link[mlr3:mlr_reflections]{mlr_reflections$learner_predict_types}} for available values.}

\item{\code{jittable}}{(\code{logical(1)})\cr
Whether the model can be jit-traced.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-LearnerTorchImage-clone"></a>}}
\if{latex}{\out{\hypertarget{method-LearnerTorchImage-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{LearnerTorchImage$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
