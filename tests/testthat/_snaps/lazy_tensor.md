# print/format

    Code
      lazy_tensor()
    Output
      <ltnsr[len=0]> 

---

    Code
      as_lazy_tensor(1)
    Output
      <ltnsr[len=1, shapes=()]> 

---

    Code
      as_lazy_tensor(matrix(1:10, ncol = 1))
    Output
      <ltnsr[len=10, shapes=(1)]> 

---

    Code
      as_lazy_tensor(ds, dataset_shapes = list(x = NULL))
    Output
      <ltnsr[len=10, shapes=unknown]> 

