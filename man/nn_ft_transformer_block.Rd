% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/PipeOpTorchFTTransformerBlock.R
\name{nn_ft_transformer_block}
\alias{nn_ft_transformer_block}
\title{Single Transformer Block for FT-Transformer}
\usage{
nn_ft_transformer_block(
  d_token,
  attention_n_heads,
  attention_dropout,
  attention_initialization,
  ffn_d_hidden = NULL,
  ffn_d_hidden_multiplier = NULL,
  ffn_dropout,
  ffn_activation,
  residual_dropout,
  prenormalization,
  is_first_layer,
  attention_normalization,
  ffn_normalization,
  query_idx = NULL,
  attention_bias,
  ffn_bias_first,
  ffn_bias_second
)
}
\arguments{
\item{d_token}{(\code{integer(1)})\cr
The dimension of the embedding.}

\item{attention_n_heads}{(\code{integer(1)})\cr
Number of attention heads.}

\item{attention_dropout}{(\code{numeric(1)})\cr
Dropout probability in the attention mechanism.}

\item{attention_initialization}{(\code{character(1)})\cr
Initialization method for attention weights. Either "kaiming" or "xavier".}

\item{ffn_d_hidden}{(\code{integer(1)})\cr
Hidden dimension of the feed-forward network. Multiplied by 2 if using ReGLU or GeGLU activation.}

\item{ffn_d_hidden_multiplier}{(\code{numeric(1)})\cr
Alternative way to specify the hidden dimension of the feed-forward network as \code{d_token * d_hidden_multiplier}. Also multiplied by 2 if using RegLU or GeGLU activation.}

\item{ffn_dropout}{(\code{numeric(1)})\cr
Dropout probability in the feed-forward network.}

\item{ffn_activation}{(\code{nn_module})\cr
Activation function for the feed-forward network. Default value is \code{nn_reglu}.}

\item{residual_dropout}{(\code{numeric(1)})\cr
Dropout probability for residual connections.}

\item{prenormalization}{(\code{logical(1)})\cr
Whether to apply normalization before attention and FFN (\code{TRUE}) or after (\code{TRUE}).}

\item{is_first_layer}{(\code{logical(1)})\cr
Whether this is the first layer in the transformer stack. Default value is \code{FALSE}.}

\item{attention_normalization}{(\code{nn_module})\cr
Normalization module to use for attention. Default value is \code{nn_layer_norm}.}

\item{ffn_normalization}{(\code{nn_module})\cr
Normalization module to use for the feed-forward network. Default value is \code{nn_layer_norm}.}

\item{query_idx}{(\code{integer()} or \code{NULL})\cr
Indices of the tensor to apply attention to. Should not be set manually.
If NULL, then attention is applied to the entire tensor.
In the last block in a stack of transformers, this is set to \code{-1}
so that attention is applied only to the embedding of the CLS token.}

\item{attention_bias}{(\code{logical(1)})\cr
Whether attention has a bias. Default is \code{TRUE}}

\item{ffn_bias_first}{(\code{logical(1)})\cr
Whether the first layer in the FFN has a bias. Default is \code{TRUE}}

\item{ffn_bias_second}{(\code{logical(1)})\cr
Whether the second layer in the FFN has a bias. Default is \code{TRUE}}
}
\description{
A transformer block consisting of a multi-head self-attention mechanism followed by a feed-forward
network.

This is used in \code{\link{LearnerTorchFTTransformer}}.
}
\references{
Devlin, Jacob, Chang, Ming-Wei, Lee, Kenton, Toutanova, Kristina (2018).
\dQuote{Bert: Pre-training of deep bidirectional transformers for language understanding.}
\emph{arXiv preprint arXiv:1810.04805}.
Gorishniy Y, Rubachev I, Khrulkov V, Babenko A (2021).
\dQuote{Revisiting Deep Learning  for Tabular Data.}
\emph{arXiv}, \bold{2106.11959}.
}
