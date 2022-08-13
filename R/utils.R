

inferps = function(fn, ignore = character(0)) {
  assert_function(fn)
  assert_character(ignore, any.missing = FALSE)
  frm = formals(fn)
  frm = frm[names(frm) %nin% ignore]

  frm_domains = lapply(frm, function(formal) {
    switch(typeof(formal),
      logical = p_lgl(),
      integer = p_int(),
      double = p_dbl(),
      p_uty()
    )
  })

  do.call(paradox::ps, frm_domains)
}
