# ------------------------------------------------------------------------------------------------
# SAINT (Self-Attention and Intersample Attention Transformer)
#
# Upstream repository: https://github.com/somepago/saint
# Ported from commit:  e288e84c77a54cfd2ffb55a53678fb7cbbb16630
# Upstream license:    Apache License 2.0 (see inst/COPYRIGHTS)
#
# The port covers the *supervised* forward path only, i.e. the path that `train.py` of the
# upstream repository executes:
#
#   _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model)
#   reps   = model.transformer(x_categ_enc, x_cont_enc)
#   y_reps = reps[:, 0, :]
#   y_outs = model.mlpfory(y_reps)
#
# Intentional deviations from upstream (all deliberate, none of them change the numerics of the
# supervised forward pass):
#
# * Out of scope / not ported: self-supervised pre-training (`pretraining.py`, `SAINT_pretrain`),
#   the mask / imputation embeddings (`mask_embeds_cat`, `mask_embeds_cont`, `single_mask`,
#   `cat_mask_offset`, `con_mask_offset` and the `cat_mask` / `con_mask` machinery of
#   `augmentations.embed_data_mask()`), mixup and CutMix augmentation, the contrastive and
#   denoising losses, the vision variants (`pretrainmodel_vision.py`, the `vision_dset`
#   positional encodings) and the `justmlp` / `attn` / `attnmlp` attention types.
#   In `train.py` all masks are all-ones, which makes the mask embeddings inert, so dropping them
#   leaves the supervised forward pass bit-identical.
# * Out of scope: `cont_embeddings` modes other than `'MLP'` (`'Noemb'`, `'pos_singleMLP'`).
#   Only `'MLP'` is ported, which is the upstream default and the only mode `embed_data_mask()`
#   supports.
# * Not ported: the heads that the supervised path never touches (`self.mlp`, `self.mlp1`,
#   `self.mlp2`, `self.pt_mlp`, `self.pt_mlp2`, `self.norm`, `self.pos_encodings`) and the unused
#   `self.embeds` / `self.mask_embed` submodules of `RowColTransformer`. They contribute
#   parameters to upstream's `state_dict()` but never receive a gradient in supervised training.
# * CLS token: upstream fakes a CLS token by prepending an extra *categorical column* of constant
#   value 0 with cardinality 1 to the data (`data_openml.py`, `cat_dims = np.append([1], cat_dims)`
#   and `DataSetCatCon.cls`). Because that column is constant, its embedding is always row 0 of
#   `model.embeds`. Here the CLS token is an explicit learned parameter (`self$cls_token`) instead.
#   This is numerically equivalent (verified against the reference implementation) and avoids
#   having to inject a fake column into the data.
# * Indexing: R's `nn_embedding()` is 1-based and `batchgetter_categ()` produces 1-based codes for
#   `factor` / `ordered` features (but 0-based codes for `logical` features, because
#   `as.integer()` maps `FALSE`/`TRUE` to 0/1). The cumulative offsets are therefore computed as
#   `cumsum(c(0, cardinalities[-n])) + (1 - min_code)` rather than copied from upstream's 0-based
#   `categories_offset`.
# * `attn_dropout` is accepted (so that upstream configurations can be transferred) but, exactly
#   as upstream, it has **no effect**: `models/model.py::Attention.__init__` creates
#   `self.dropout = nn.Dropout(dropout)` and `Attention.forward()` never uses it. We reproduce
#   this behaviour rather than "fixing" it, so that results match the published implementation.
# * Upstream's `PreNorm(dim, Residual(fn))` computes `fn(LayerNorm(x)) + LayerNorm(x)`, i.e. the
#   residual branch is added to the *normalised* input, not to the raw input. This is unusual but
#   faithfully reproduced by `nn_saint_prenorm_residual`.
# * The hidden widths `100` (continuous feature embedding MLP) and `1000` (output head) are
#   hardcoded upstream; here they are exposed as `d_hidden_num_embed` and `d_hidden_head` with
#   these values as defaults. Likewise, `RowColTransformer` hardcodes `dim_head = 64` for the
#   intersample attention; this is exposed as `dim_head_row` with default 64.
# * The feed-forward multiplier (`mult = 4` in `FeedForward`) is hardcoded, as upstream.
# * Not ported: `train.py` silently overrides the user's hyperparameters for non-`'col'` attention
#   (`transformer_depth = 1`, `attention_heads = min(4, heads)`, `attention_dropout = 0.8`,
#   `ff_dropout = 0.8`, `embedding_size = min(32, d)`) and for wide datasets. mlr3torch respects
#   the configured values instead; the upstream numbers are a reasonable starting point for tuning.
# * Not ported: upstream standardizes the numeric features with the training mean / standard
#   deviation inside `DataSetCatCon`. In mlr3 this belongs into a preprocessing pipeline
#   (`po("scale")`), so it is not part of the network.
# * Shape handling: upstream uses `einops.rearrange()` with explicit sizes. Here the equivalent
#   reshapes are written so that the batch size stays symbolic (`unflatten()` / `flatten()` /
#   `reshape(-1, ...)` / `expand_as()`) so that the network can be traced with `jit_trace()` and
#   reused with a different batch size (`jittable = TRUE`). This does not change the numerics --
#   the equivalence check against PyTorch is exact for all three attention types.
# ------------------------------------------------------------------------------------------------

# GEGLU (models/model.py::GEGLU)
nn_saint_geglu = nn_module("nn_saint_geglu",
  initialize = function() NULL,
  forward = function(input) {
    chunks = input$chunk(2L, dim = -1L)
    chunks[[1L]] * nnf_gelu(chunks[[2L]])
  }
)

# FeedForward (models/model.py::FeedForward), mult is hardcoded to 4 upstream
nn_saint_feed_forward = nn_module("nn_saint_feed_forward",
  initialize = function(dim, mult = 4L, dropout = 0) {
    self$linear1 = nn_linear(dim, dim * mult * 2L)
    self$geglu = nn_saint_geglu()
    self$dropout = nn_dropout(dropout)
    self$linear2 = nn_linear(dim * mult, dim)
  },
  forward = function(input) {
    x = self$linear1(input)
    x = self$geglu(x)
    x = self$dropout(x)
    self$linear2(x)
  }
)

# Attention (models/model.py::Attention)
# NOTE: upstream constructs an `nn.Dropout(dropout)` here but never applies it; `dropout` is
# therefore accepted and ignored, see the header comment.
nn_saint_attention = nn_module("nn_saint_attention",
  initialize = function(dim, heads = 8L, dim_head = 16L, dropout = 0) {
    inner_dim = dim_head * heads
    self$heads = heads
    self$dim_head = dim_head
    self$scale = dim_head^(-0.5)
    self$to_qkv = nn_linear(dim, inner_dim * 3L, bias = FALSE)
    self$to_out = nn_linear(inner_dim, dim)
  },
  forward = function(input) {
    h = self$heads
    d = self$dim_head
    qkv = self$to_qkv(input)$chunk(3L, dim = -1L)
    # rearrange(t, "b n (h d) -> b h n d"); `unflatten()` keeps the leading dimensions symbolic so
    # that the module can be traced with `jit_trace()` and reused with other batch sizes.
    split_heads = function(t) t$unflatten(-1L, c(h, d))$permute(c(1L, 3L, 2L, 4L))
    q = split_heads(qkv[[1L]])
    k = split_heads(qkv[[2L]])
    v = split_heads(qkv[[3L]])
    sim = torch_matmul(q, k$transpose(-1L, -2L)) * self$scale
    attn = nnf_softmax(sim, dim = -1L)
    out = torch_matmul(attn, v)
    # rearrange(out, "b h n d -> b n (h d)")
    out = out$permute(c(1L, 3L, 2L, 4L))$flatten(start_dim = 3L)
    self$to_out(out)
  }
)

# PreNorm(dim, Residual(fn)) (models/model.py::PreNorm, models/model.py::Residual)
# Careful: this is `fn(norm(x)) + norm(x)`, not `fn(norm(x)) + x`.
nn_saint_prenorm_residual = nn_module("nn_saint_prenorm_residual",
  initialize = function(dim, fn) {
    self$norm = nn_layer_norm(dim)
    self$fn = fn
  },
  forward = function(input) {
    x = self$norm(input)
    self$fn(x) + x
  }
)

# simple_MLP (models/model.py::simple_MLP)
nn_saint_simple_mlp = nn_module("nn_saint_simple_mlp",
  initialize = function(d_in, d_hidden, d_out) {
    self$linear1 = nn_linear(d_in, d_hidden)
    self$relu = nn_relu()
    self$linear2 = nn_linear(d_hidden, d_out)
  },
  forward = function(input) {
    self$linear2(self$relu(self$linear1(input)))
  }
)

# Transformer (models/model.py::Transformer): column (feature) attention only.
# The `depth` blocks are stored flat as attn_1, ff_1, attn_2, ff_2, ...
nn_saint_transformer = nn_module("nn_saint_transformer",
  initialize = function(dim, depth, heads, dim_head, attn_dropout, ff_dropout) {
    self$depth = depth
    self$layers = nn_module_list(unlist(lapply(seq_len(depth), function(i) {
      list(
        nn_saint_prenorm_residual(dim,
          nn_saint_attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
        nn_saint_prenorm_residual(dim,
          nn_saint_feed_forward(dim, dropout = ff_dropout))
      )
    }), recursive = FALSE))
  },
  forward = function(input) {
    x = input
    for (i in seq_len(self$depth)) {
      x = self$layers[[2L * i - 1L]](x)
      x = self$layers[[2L * i]](x)
    }
    x
  }
)

# RowColTransformer (models/model.py::RowColTransformer): intersample ("row") attention, either
# alone (`style = "row"`) or interleaved with column attention (`style = "colrow"`).
# The row attention flattens the whole batch into a single sequence
# (`rearrange(x, "b n d -> 1 b (n d)")`), which is what makes a row's representation depend on the
# other rows of its batch.
nn_saint_rowcol_transformer = nn_module("nn_saint_rowcol_transformer",
  initialize = function(dim, nfeats, depth, heads, dim_head, dim_head_row, attn_dropout,
    ff_dropout, style = "colrow") {
    self$style = style
    self$depth = depth
    self$nfeats = nfeats
    self$dim = dim
    dim_row = dim * nfeats
    blocks = lapply(seq_len(depth), function(i) {
      row_blocks = list(
        nn_saint_prenorm_residual(dim_row,
          nn_saint_attention(dim_row, heads = heads, dim_head = dim_head_row, dropout = attn_dropout)),
        nn_saint_prenorm_residual(dim_row,
          nn_saint_feed_forward(dim_row, dropout = ff_dropout))
      )
      if (identical(style, "colrow")) {
        c(list(
          nn_saint_prenorm_residual(dim,
            nn_saint_attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
          nn_saint_prenorm_residual(dim,
            nn_saint_feed_forward(dim, dropout = ff_dropout))
        ), row_blocks)
      } else {
        row_blocks
      }
    })
    self$n_per_layer = if (identical(style, "colrow")) 4L else 2L
    self$layers = nn_module_list(unlist(blocks, recursive = FALSE))
  },
  forward = function(input) {
    x = input
    n = self$nfeats
    d = self$dim
    k = self$n_per_layer
    for (i in seq_len(self$depth)) {
      off = (i - 1L) * k
      if (k == 4L) {
        x = self$layers[[off + 1L]](x)
        x = self$layers[[off + 2L]](x)
      }
      # rearrange(x, "b n d -> 1 b (n d)"); `-1` keeps the batch size symbolic under `jit_trace()`
      x = x$reshape(c(1L, -1L, n * d))
      x = self$layers[[off + k - 1L]](x)
      x = self$layers[[off + k]](x)
      # rearrange(x, "1 b (n d) -> b n d")
      x = x$reshape(c(-1L, n, d))
    }
    x
  }
)

#' @title SAINT Network
#'
#' @name nn_saint
#'
#' @description
#' The SAINT network for tabular data (Somepalli et al., 2021), ported from the official
#' implementation (<https://github.com/somepago/saint>).
#'
#' Each numeric feature is embedded with its own two-layer MLP, each categorical feature is
#' embedded with a shared embedding table (using cumulative offsets), and a learned `CLS` token is
#' prepended to the resulting token sequence.
#' The token sequence is processed by a transformer that -- depending on `attention_type` --
#' applies self-attention over the *features* of a row (`"col"`), over the *rows* of a batch
#' (`"row"`, "intersample attention"), or both (`"colrow"`).
#' The final prediction is obtained by applying a two-layer MLP to the `CLS` token.
#'
#' Either `task` or (`n_features_num`, `cardinalities`, `d_out`) must be provided.
#'
#' @section Intersample attention:
#' For `attention_type` `"row"` and `"colrow"`, the representation of a row depends on the other
#' rows in the same batch. See the *Intersample Attention* section of [`LearnerTorchSAINT`].
#'
#' @param task ([`Task`][mlr3::Task] or `NULL`)\cr
#'   Task from which the number of numeric features, the cardinalities of the categorical features
#'   and the output dimension are inferred.
#' @param n_features_num (`integer(1)` or `NULL`)\cr
#'   Number of numeric features. Inferred from `task` if `NULL`.
#' @param cardinalities (`integer()` or `NULL`)\cr
#'   Number of categories for each categorical feature, in the order in which the columns appear in
#'   `x_cat`. Inferred from `task` if `NULL`.
#' @param categ_min_code (`integer()`)\cr
#'   Smallest integer code that occurs for each categorical feature. This is `1` for `factor` and
#'   `ordered` features and `0` for `logical` features, because [`batchgetter_categ()`] applies
#'   `as.integer()`. Recycled if of length 1. Ignored when `task` is given.
#' @param d_out (`integer(1)` or `NULL`)\cr
#'   Output dimension. Inferred from `task` via [`output_dim_for()`] if `NULL`.
#' @param attention_type (`character(1)`)\cr
#'   One of `"col"` (feature self-attention only), `"row"` (intersample attention only) or
#'   `"colrow"` (both, the default and the variant recommended by the paper).
#' @param d_token (`integer(1)`)\cr
#'   Embedding dimension of a token. Default `32`.
#' @param depth (`integer(1)`)\cr
#'   Number of transformer blocks. Default `6`.
#' @param n_heads (`integer(1)`)\cr
#'   Number of attention heads. Default `8`.
#' @param dim_head (`integer(1)`)\cr
#'   Dimension of an attention head for the column (feature) attention. Default `16`.
#' @param dim_head_row (`integer(1)`)\cr
#'   Dimension of an attention head for the intersample (row) attention. Hardcoded to `64`
#'   upstream, which is also the default here.
#' @param attn_dropout (`numeric(1)`)\cr
#'   Attention dropout rate. **Has no effect**: the upstream `Attention` module creates a dropout
#'   layer but never applies it. The parameter exists so that upstream configurations can be
#'   transferred unchanged. Default `0.1`.
#' @param ff_dropout (`numeric(1)`)\cr
#'   Dropout rate in the feed-forward blocks. Default `0.1`.
#' @param d_hidden_num_embed (`integer(1)`)\cr
#'   Hidden width of the per-feature MLP that embeds a numeric feature. Hardcoded to `100`
#'   upstream, which is also the default here.
#' @param d_hidden_head (`integer(1)`)\cr
#'   Hidden width of the output head. Hardcoded to `1000` upstream, which is also the default here.
#'
#' @return An [`nn_module`][torch::nn_module] whose `forward(x_num, x_cat)` method returns a tensor
#'   of shape `(batch, d_out)`. Both arguments default to `NULL`, so tasks with only numeric or
#'   only categorical features are supported.
#'
#' @references
#' `r format_bib("somepalli2021saint")`
#'
#' @export
nn_saint = nn_module("nn_saint",
  initialize = function(task = NULL, n_features_num = NULL, cardinalities = NULL,
    categ_min_code = 1L, d_out = NULL, attention_type = "colrow", d_token = 32L, depth = 6L,
    n_heads = 8L, dim_head = 16L, dim_head_row = 64L, attn_dropout = 0.1, ff_dropout = 0.1,
    d_hidden_num_embed = 100L, d_hidden_head = 1000L) {
    if (!is.null(task)) {
      assert_class(task, "Task")
      n_features_num = n_num_features(task)
      info = saint_categ_info(task)
      cardinalities = info$cardinalities
      categ_min_code = info$min_code
      d_out = output_dim_for(task)
    }
    n_features_num = assert_int(n_features_num, lower = 0L, coerce = TRUE)
    cardinalities = assert_integerish(cardinalities %??% integer(0), lower = 1L,
      any.missing = FALSE, coerce = TRUE)
    categ_min_code = assert_integerish(categ_min_code, any.missing = FALSE, coerce = TRUE)
    if (length(categ_min_code) == 1L) {
      categ_min_code = rep(categ_min_code, length(cardinalities))
    }
    assert_true(length(categ_min_code) == length(cardinalities))
    d_out = assert_int(d_out, lower = 1L, coerce = TRUE)
    attention_type = assert_choice(attention_type, c("col", "row", "colrow"))
    d_token = assert_int(d_token, lower = 1L, coerce = TRUE)
    depth = assert_int(depth, lower = 0L, coerce = TRUE)
    n_heads = assert_int(n_heads, lower = 1L, coerce = TRUE)
    dim_head = assert_int(dim_head, lower = 1L, coerce = TRUE)
    dim_head_row = assert_int(dim_head_row, lower = 1L, coerce = TRUE)
    d_hidden_num_embed = assert_int(d_hidden_num_embed, lower = 1L, coerce = TRUE)
    d_hidden_head = assert_int(d_hidden_head, lower = 1L, coerce = TRUE)
    assert_number(attn_dropout, lower = 0, upper = 1)
    assert_number(ff_dropout, lower = 0, upper = 1)

    n_features_categ = length(cardinalities)
    if (n_features_num + n_features_categ == 0L) {
      stopf("nn_saint needs at least one feature.")
    }

    self$n_features_num = n_features_num
    self$n_features_categ = n_features_categ
    self$d_token = d_token
    self$attention_type = attention_type

    # CLS token, cf. header comment: upstream emulates this with a constant categorical column of
    # cardinality 1, whose embedding is row 0 of `model.embeds`.
    self$cls_token = nn_parameter(nn_init_normal_(torch_empty(1L, 1L, d_token)))

    if (n_features_categ > 0L) {
      # 1-based cumulative offsets, see header comment
      offsets = cumsum(c(0L, cardinalities[-n_features_categ])) + (1L - categ_min_code)
      self$categ_offsets = nn_buffer(torch_tensor(matrix(offsets, nrow = 1L), dtype = torch_long()))
      self$categ_embeddings = nn_embedding(sum(cardinalities), d_token)
    }
    if (n_features_num > 0L) {
      # upstream: nn.ModuleList([simple_MLP([1, 100, dim]) for _ in range(num_continuous)])
      self$num_embeddings = nn_module_list(lapply(seq_len(n_features_num), function(i) {
        nn_saint_simple_mlp(1L, d_hidden_num_embed, d_token)
      }))
    }

    # +1 for the CLS token; upstream counts it because it is part of `num_categories`
    nfeats = 1L + n_features_categ + n_features_num

    self$transformer = if (attention_type == "col") {
      nn_saint_transformer(
        dim = d_token, depth = depth, heads = n_heads, dim_head = dim_head,
        attn_dropout = attn_dropout, ff_dropout = ff_dropout
      )
    } else {
      nn_saint_rowcol_transformer(
        dim = d_token, nfeats = nfeats, depth = depth, heads = n_heads, dim_head = dim_head,
        dim_head_row = dim_head_row, attn_dropout = attn_dropout, ff_dropout = ff_dropout,
        style = attention_type
      )
    }

    # upstream: self.mlpfory = simple_MLP([dim, 1000, y_dim])
    self$head = nn_saint_simple_mlp(d_token, d_hidden_head, d_out)
  },
  forward = function(x_num = NULL, x_cat = NULL) {
    # When a task has only one input tensor, mlr3torch calls the network *by position*
    # (see `learner_torch_train()`), so a purely categorical task arrives in `x_num`.
    if (self$n_features_num == 0L && is.null(x_cat)) {
      x_cat = x_num
      x_num = NULL
    }
    tokens = list()
    if (self$n_features_categ > 0L) {
      tokens[[length(tokens) + 1L]] = self$categ_embeddings(x_cat + self$categ_offsets)
    }
    if (self$n_features_num > 0L) {
      num_tokens = lapply(seq_len(self$n_features_num), function(i) {
        # `[` rather than `$narrow()`: `narrow()` does not trace correctly with `jit_trace()`
        self$num_embeddings[[i]](x_num[, i, drop = FALSE])
      })
      tokens[[length(tokens) + 1L]] = torch_stack(num_tokens, dim = 2L)
    }
    x = if (length(tokens) == 1L) tokens[[1L]] else torch_cat(tokens, dim = 2L)
    # prepend the CLS token; `expand_as()` derives the batch size from the feature tokens, which
    # keeps it symbolic under `jit_trace()`
    x = torch_cat(list(self$cls_token$expand_as(x[, 1L, , drop = FALSE]), x), dim = 2L)
    x = self$transformer(x)
    # upstream: y_reps = reps[:, 0, :]; y_outs = model.mlpfory(y_reps)
    self$head(x[, 1L, ])
  }
)

# cardinalities and smallest integer code per categorical feature, in the column order produced by
# `ingress_categ()`
saint_categ_info = function(task) {
  types = task$feature_types
  categ = types[get("type") %in% c("factor", "ordered", "logical"), ]
  cardinalities = integer(nrow(categ))
  min_code = integer(nrow(categ))
  levs = task$levels(categ$id)
  for (i in seq_len(nrow(categ))) {
    if (categ$type[i] == "logical") {
      # `batchgetter_categ()` encodes logicals via `as.integer()`, i.e. as 0 / 1
      cardinalities[i] = 2L
      min_code[i] = 0L
    } else {
      cardinalities[i] = length(levs[[categ$id[i]]])
      min_code[i] = 1L
    }
  }
  list(cardinalities = cardinalities, min_code = min_code)
}

#' @title SAINT
#'
#' @templateVar name saint
#' @templateVar task_types classif, regr
#' @templateVar param_vals d_token = 8, depth = 1, n_heads = 2, dim_head = 4
#' @template params_learner
#' @template learner
#' @template learner_example
#'
#' @description
#' SAINT (Self-Attention and Intersample Attention Transformer) for tabular data.
#'
#' Numeric features are embedded with a per-feature MLP, categorical features with an embedding
#' table, and a learned `CLS` token is prepended to the token sequence.
#' A transformer then applies attention *across features* (like a standard tabular transformer) and
#' -- this is SAINT's key idea -- *across the rows of a batch* ("intersample attention"), which lets
#' a row borrow information from other rows.
#' The `CLS` token of the last layer is passed through a two-layer MLP to obtain the prediction.
#'
#' This is a port of the supervised path of the official implementation
#' (<https://github.com/somepago/saint>, Apache-2.0). The self-supervised pre-training stage of the
#' paper (contrastive + denoising losses, CutMix / mixup augmentation) is **not** implemented.
#'
#' @section Intersample Attention:
#' With `attention_type` set to `"row"` or `"colrow"` (the default), SAINT attends over the rows of
#' the current batch. This has consequences that users must be aware of:
#'
#' * The prediction for an observation **depends on which other observations are in its batch**.
#'   Predictions therefore change when `batch_size` changes and when the row order changes.
#' * A batch containing a single row degenerates: attention over one row reduces to a pass-through.
#'   This can happen at prediction time for the last (incomplete) batch. Prefer a `batch_size` that
#'   divides the number of rows to predict, or use `attention_type = "col"` if you need
#'   row-independent predictions.
#' * The learner warns at prediction time when `batch_size` is smaller than 8 and intersample
#'   attention is active.
#'
#' This is inherent to the architecture as published, not an implementation artefact. With
#' `attention_type = "col"` the network reduces to a standard feature-wise transformer and
#' predictions are independent across rows.
#'
#' @section Parameters:
#' Parameters from [`LearnerTorch`], as well as:
#' * `attention_type` :: `character(1)`\cr
#'   One of `"col"`, `"row"` or `"colrow"`. Initialized to `"colrow"`.
#' * `d_token` :: `integer(1)`\cr
#'   Embedding dimension of a token. Initialized to `32`.
#' * `depth` :: `integer(1)`\cr
#'   Number of transformer blocks. Initialized to `6`.
#' * `n_heads` :: `integer(1)`\cr
#'   Number of attention heads. Initialized to `8`.
#' * `dim_head` :: `integer(1)`\cr
#'   Dimension of an attention head of the feature ("col") attention. Initialized to `16`.
#' * `dim_head_row` :: `integer(1)`\cr
#'   Dimension of an attention head of the intersample ("row") attention. Hardcoded to `64` in the
#'   reference implementation; initialized to `64`.
#' * `attn_dropout` :: `numeric(1)`\cr
#'   Attention dropout. Initialized to `0.1`. **This has no effect**, mirroring the reference
#'   implementation, which constructs but never applies this dropout layer.
#' * `ff_dropout` :: `numeric(1)`\cr
#'   Dropout of the feed-forward blocks. Initialized to `0.1`.
#' * `d_hidden_num_embed` :: `integer(1)`\cr
#'   Hidden width of the MLP embedding a numeric feature. Hardcoded to `100` in the reference
#'   implementation; initialized to `100`.
#' * `d_hidden_head` :: `integer(1)`\cr
#'   Hidden width of the output head. Hardcoded to `1000` in the reference implementation;
#'   initialized to `1000`.
#'
#' @section Preprocessing:
#' The reference implementation standardizes the numeric features using the training mean and
#' standard deviation. This is not done automatically here; consider `po("scale")`.
#'
#' @references
#' `r format_bib("somepalli2021saint")`
#' @export
LearnerTorchSAINT = R6Class("LearnerTorchSAINT",
  inherit = LearnerTorch,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(task_type, optimizer = NULL, loss = NULL, callbacks = list()) {
      private$.param_set_base = ps(
        attention_type     = p_fct(levels = c("col", "row", "colrow"), init = "colrow", tags = "train"),
        d_token            = p_int(lower = 1L, init = 32L, tags = "train"),
        depth              = p_int(lower = 0L, init = 6L, tags = "train"),
        n_heads            = p_int(lower = 1L, init = 8L, tags = "train"),
        dim_head           = p_int(lower = 1L, init = 16L, tags = "train"),
        dim_head_row       = p_int(lower = 1L, init = 64L, tags = "train"),
        attn_dropout       = p_dbl(lower = 0, upper = 1, init = 0.1, tags = "train"),
        ff_dropout         = p_dbl(lower = 0, upper = 1, init = 0.1, tags = "train"),
        d_hidden_num_embed = p_int(lower = 1L, init = 100L, tags = "train"),
        d_hidden_head      = p_int(lower = 1L, init = 1000L, tags = "train")
      )

      super$initialize(
        task_type = task_type,
        id = paste0(task_type, ".saint"),
        label = "SAINT",
        param_set = alist(private$.param_set_base),
        optimizer = optimizer,
        callbacks = callbacks,
        loss = loss,
        man = "mlr3torch::mlr_learners.saint",
        feature_types = c("numeric", "integer", "logical", "factor", "ordered"),
        # Verified empirically for all three attention types (and for ragged final batches):
        # the traced module reproduces the eager module exactly for batch sizes other than the
        # one it was traced with. This requires all reshapes to keep the batch size symbolic,
        # see the comments in `nn_saint_attention` / `nn_saint_rowcol_transformer` / `nn_saint`.
        jittable = TRUE
      )
    }
  ),
  private = list(
    .ingress_tokens = function(task, param_vals) {
      n_num = n_num_features(task)
      n_categ = n_categ_features(task)
      out = list()
      if (n_num > 0L) {
        out$x_num = ingress_num(shape = c(NA, n_num))
      }
      if (n_categ > 0L) {
        out$x_cat = ingress_categ(shape = c(NA, n_categ))
      }
      out
    },
    .network = function(task, param_vals) {
      args = param_vals[intersect(names(param_vals), formalArgs(nn_saint))]
      invoke(nn_saint, task = task, .args = args)
    },
    .check_predict_task = function(task, param_vals) {
      # `attention_type` is a train-only parameter, so take it from the trained network
      network = self$model$network
      attention_type = if (inherits(network, "nn_saint")) {
        network$attention_type
      } else {
        self$param_set$values$attention_type
      }
      if (!is.null(attention_type) && attention_type != "col" && isTRUE(param_vals$batch_size < 8L)) {
        warningf("Learner '%s' uses intersample attention (attention_type = '%s'), so predictions depend on the other rows in the same batch. The configured batch_size (%i) is very small, which makes predictions unstable.", # nolint
          self$id, attention_type, param_vals$batch_size)
      }
      TRUE
    }
  )
)

#' @include aaa.R
register_learner("classif.saint", LearnerTorchSAINT)
register_learner("regr.saint", LearnerTorchSAINT)
