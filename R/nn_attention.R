
#' @references
#' `r format_bib("gorishniy2021revisiting", "devlin2018bert")`
#' @export
nn_attention = nn_module("nn_attention",
  initialize = function(d_token, n_heads = 1L, dropout = 0.5, bias = TRUE,
    initialization = "kaiming") {
    n_heads = assert_int(n_heads)
    d_token = assert_int(d_token)
    dropout = assert_numeric(dropout, lower = 0, upper = 1, len = 1, any.missing = FALSE)
    bias = assert_flag(bias)
    assert_choice(initialization, c("kaiming", "xavier"))
    if (n_heads > 1L) {
      assert(d_token %% n_heads == 0L)
    }
    self$W_q = nn_linear(d_token, d_token, bias)
    self$W_k = nn_linear(d_token, d_token, bias)
    self$W_v = nn_linear(d_token, d_token, bias)
    self$W_out = if (n_heads > 1) {
      nn_linear(d_token, d_token, bias)
    } else {
      NULL
    }
    self$n_heads = n_heads
    self$dropout = if (dropout) {
      nn_dropout(dropout)
    } else {
      NULL
    }
    for (name in c("W_q", "W_k", "W_v")) {
      m = self[[name]]
      if (initialization == "xavier" && (name != "W_v" || is.null(self$W_out))) {
        nn_init_xavier_uniform_(m$weight, gain = 1 / sqrt(2))
      }
      if (!is.null(m$bias)) {
        nn_init_zeros_(m$bias)
      }
    }
    if (!is.null(self$W_out) && bias) {
      nn_init_zeros_(self$W_out$bias)
    }
  },
  reshape = function(x) {
    batch_size = x$shape[[1L]]
    n_tokens = x$shape[[2L]]
    d_tokens = x$shape[[3L]]

    d_head = d_tokens %/% self$n_heads

    x = x$reshape(c(batch_size, n_tokens, self$n_heads, d_head))
    x = x$transpose(2L, 3L)
    x = x$reshape(c(batch_size * self$n_heads, n_tokens, d_head))
    return(x)
  },
  forward = function(query, key) {
    x_q = query
    x_kv = key
    q = self$W_q(x_q)
    k = self$W_k(x_kv)
    v = self$W_v(x_kv)
    # TODO: This should be moved somewhere else?
    for (tensor in list(q, k, v)) {
      assert_true(tensor$shape[length(tensor$shape)] %% self$n_heads == 0L)
    }
    batch_size = nrow(q)
    d_head_key = k$shape[[length(k$shape)]] # self.n_heads
    d_head_value = v$shape[[length(v$shape)]] # self.n_heads
    n_q_tokens = q$shape[[2L]]
    q = self$reshape(q)
    k = self$reshape(k)
    attention_logits = torch_matmul(q, k$transpose(2L, 3L) / sqrt(d_head_key))
    attention_probs = nnf_softmax(attention_logits, dim = -1L)
    if (!is.null(self$dropout)) {
      attention_probs = self$dropout(attention_probs)
    }
    x = torch_matmul(attention_probs, self$reshape(v))
    x = x$reshape(c(batch_size, self$n_heads, n_q_tokens, d_head_value))
    x = x$transpose(2L, 3L)
    x = x$reshape(c(batch_size, n_q_tokens, self$n_heads * d_head_value))
    if (!is.null(self$W_out)) {
      x = self$W_out
    }
    # output = list(x = x, attention = list(logits = attention_logits, probs = attention_probs))
    # return(output)
    return(x)
  }
)

nn_self_attention = nn_module(
  inherit = nn_attention,
  initialize = function(d_token, n_heads = 1L, dropout = 0.5, bias = TRUE) {
    super$initialize(d_token, n_heads, dropout, bias)
  },
  forward = function(input) {
    super$forward(query = input, key = input)
  }
)
