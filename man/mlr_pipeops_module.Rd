% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PipeOpModule.R
\name{mlr_pipeops_module}
\alias{mlr_pipeops_module}
\alias{PipeOpModule}
\title{Class for Torch Module Wrappers}
\description{
\code{PipeOpModule} wraps an \code{\link[torch:nn_module]{nn_module}} or \code{function} that is being called during the \code{train} phase of this
\code{\link[mlr3pipelines:PipeOp]{mlr3pipelines::PipeOp}}. By doing so, this allows to assemble \code{PipeOpModule}s in a computational
\code{\link[mlr3pipelines:Graph]{mlr3pipelines::Graph}} that represents either a neural network or a preprocessing graph of a \code{\link{lazy_tensor}}.
In most cases it is easier to create such a network by creating a graph that generates this graph.

In most cases it is easier to create such a network by creating a structurally related graph consisting
of nodes of class \code{\link{PipeOpTorchIngress}} and \code{\link{PipeOpTorch}}. This graph will then generate the graph consisting
of \code{PipeOpModule}s as part of the \code{\link{ModelDescriptor}}.
}
\section{Input and Output Channels}{

The number and names of the input and output channels can be set during construction. They input and output
\code{"torch_tensor"} during training, and \code{NULL} during prediction as the prediction phase currently serves no
meaningful purpose.
}

\section{State}{

The state is the value calculated by the public method \code{shapes_out()}.
}

\section{Parameters}{

No parameters.
}

\section{Internals}{

During training, the wrapped \code{\link[torch:nn_module]{nn_module}} / \code{function} is called with the provided inputs in the order in which
the channels are defined. Arguments are \strong{not} matched by name.
}

\examples{
\dontshow{if (torch::torch_is_installed()) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
## creating an PipeOpModule manually

# one input and output channel
po_module = po("module",
  id = "linear",
  module = torch::nn_linear(10, 20),
  inname = "input",
  outname = "output"
)
x = torch::torch_randn(16, 10)
# This calls the forward function of the wrapped module.
y = po_module$train(list(input = x))
str(y)

# multiple input and output channels
nn_custom = torch::nn_module("nn_custom",
  initialize = function(in_features, out_features) {
    self$lin1 = torch::nn_linear(in_features, out_features)
    self$lin2 = torch::nn_linear(in_features, out_features)
  },
  forward = function(x, z) {
    list(out1 = self$lin1(x), out2 = torch::nnf_relu(self$lin2(z)))
  }
)

module = nn_custom(3, 2)
po_module = po("module",
  id = "custom",
  module = module,
  inname = c("x", "z"),
  outname = c("out1", "out2")
)
x = torch::torch_randn(1, 3)
z = torch::torch_randn(1, 3)
out = po_module$train(list(x = x, z = z))
str(out)

# How such a PipeOpModule is usually generated
graph = po("torch_ingress_num") \%>>\% po("nn_linear", out_features = 10L)
result = graph$train(tsk("iris"))
# The PipeOpTorchLinear generates a PipeOpModule and adds it to a new (module) graph
result[[1]]$graph
linear_module = result[[1L]]$graph$pipeops$nn_linear
linear_module
formalArgs(linear_module$module)
linear_module$input$name

# Constructing a PipeOpModule using a simple function
po_add1 = po("module",
  id = "add_one",
  module = function(x) x + 1
)
input = list(torch_tensor(1))
po_add1$train(input)$output
\dontshow{\}) # examplesIf}
}
\seealso{
Other Graph Network: 
\code{\link{ModelDescriptor}()},
\code{\link{TorchIngressToken}()},
\code{\link{mlr_learners_torch_model}},
\code{\link{mlr_pipeops_torch}},
\code{\link{mlr_pipeops_torch_ingress}},
\code{\link{mlr_pipeops_torch_ingress_categ}},
\code{\link{mlr_pipeops_torch_ingress_ltnsr}},
\code{\link{mlr_pipeops_torch_ingress_num}},
\code{\link{model_descriptor_to_learner}()},
\code{\link{model_descriptor_to_module}()},
\code{\link{model_descriptor_union}()},
\code{\link{nn_graph}()}

Other PipeOp: 
\code{\link{mlr_pipeops_torch_callbacks}},
\code{\link{mlr_pipeops_torch_optimizer}}
}
\concept{Graph Network}
\concept{PipeOp}
\section{Super class}{
\code{\link[mlr3pipelines:PipeOp]{mlr3pipelines::PipeOp}} -> \code{PipeOpModule}
}
\section{Public fields}{
\if{html}{\out{<div class="r6-fields">}}
\describe{
\item{\code{module}}{(\code{\link[torch:nn_module]{nn_module}})\cr
The torch module that is called during the training phase.}
}
\if{html}{\out{</div>}}
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-PipeOpModule-new}{\code{PipeOpModule$new()}}
\item \href{#method-PipeOpModule-clone}{\code{PipeOpModule$clone()}}
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
\if{html}{\out{<a id="method-PipeOpModule-new"></a>}}
\if{latex}{\out{\hypertarget{method-PipeOpModule-new}{}}}
\subsection{Method \code{new()}}{
Creates a new instance of this \link[R6:R6Class]{R6} class.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PipeOpModule$new(
  id = "module",
  module = nn_identity(),
  inname = "input",
  outname = "output",
  param_vals = list(),
  packages = character(0)
)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{id}}{(\code{character(1)})\cr
The id for of the new object.}

\item{\code{module}}{(\code{\link[torch:nn_module]{nn_module}} or \verb{function()})\cr
The torch module or function that is being wrapped.}

\item{\code{inname}}{(\code{character()})\cr
The names of the input channels.}

\item{\code{outname}}{(\code{character()})\cr
The names of the output channels. If this parameter has length 1, the parameter \link[torch:nn_module]{module} must
return a \link[torch:torch_tensor]{tensor}. Otherwise it must return a \code{list()} of tensors of corresponding length.}

\item{\code{param_vals}}{(named \code{list()})\cr
Parameter values to be set after construction.}

\item{\code{packages}}{(\code{character()})\cr
The R packages this object depends on.}
}
\if{html}{\out{</div>}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-PipeOpModule-clone"></a>}}
\if{latex}{\out{\hypertarget{method-PipeOpModule-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{PipeOpModule$clone(deep = FALSE)}\if{html}{\out{</div>}}
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
