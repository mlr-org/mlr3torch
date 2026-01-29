# printer

    Code
      as_lazy_tensor(1:2)
    Output
      <ltnsr[len=2, shapes=()]>

---

    Code
      lazy_tensor()
    Output
      <ltnsr[len=0]>

---

    Code
      as_lazy_tensor(ds, dataset_shapes = list(x = c(NA, 2, 3)))
    Output
      <ltnsr[len=3, shapes=(2,3)]>

---

    Code
      as_lazy_tensor(ds, dataset_shapes = list(x = NULL))
    Output
      <ltnsr[len=3, shapes=unknown]>
