# SOFORT: Fixe test_GraphArchitecture.R (irgendwie wird der merge call nicht zurückgegeben)

Nächste Schritte:
- Ziel: Präsentation
- Bis dahin:
    - Branching fertig
    - Lukas und meine Arbeit integrieren
    - Man kann eigene Architektur angeben
- Vignette: 3 Levels: TorchOp, Preconfigured, Komplett flexibel
- Vereinigen: make_dataloder für verschiedene Datentypen + Füge preprocessing options hinzu
Bis: 3.5.
- Problem mit der Nicht-kommutativität des concat mergers. Die Reihenfolge wird von
  dem topologishen Sort vorgegeben (weil wir da ja dann die inputs in die liste schreiben),
  das ist der entry point
- Naming: maybe rename parents and children to sccsrs and predecessors (like in pipelines)
- TorchOpAttention (mit key, value input) --> test dass auch mehrere input channels von
verschiedenen nodes kein problem sind. Evtl auch mit mehrerern outputs (?)


Steps for the graph architecture:
- How do I deal with the input definitions of the "mergers"
- FastAI: FitoneBatch FitOneEpoch FitAll fastai/learner.py


Long term idea:
- be able to easily reparametrize the graph: something like the code below should produce a graph
with the parameter x that is set as a.out_features and b.out_features
```r
g = paragraph(
  list(
    a = top("linear", out_features = x) %>>% top("relu"),
    b = top("linear", out_features = x)
  ),
  params = list(x, y),
  .merge = "add"
)

```

make_torch_op("linear", ps())
- Think about naming the tensor dimensions (batch, feature, token) for all TorchOp's
- This must only be done once, when the network is constructed

- [ ] Remove all the private$.operator (is now in TorchOp)
- [ ] LearnerTorchClassif muss von LearnerClassif inheriten
- [ ] LearnerTorchRegr muss von LearnerRegr inheriten
- [ ] repr for TorchOps (is printed in architecture)
wenn möglich immer mlr_reflections verwenden
