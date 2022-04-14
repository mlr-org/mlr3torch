# SOFORT: Fixe test_GraphArchitecture.R (irgendwie wird der merge call nicht zurückgegeben)


next:
1. Continue going through all test files (starting from top, came until TorchOpCustom)
2. Go through R files and remove old stuff
3. Fix rcmd check

## Questions:
- should we allow ops with more than one output? (attention logits and probs of nn_attention)

## block()
--> What is a block?
I think block() should create a TorchOpBlock and does some nice things like appending missing
top("input") and potentially also reparametrize the thing --> we need a language for that


# Transformationen:
1. Vorschlag: In den Task schreiben
2. Attribute (schreibe Subset Funktion)
Inherite von PipeOpTaskPreproc


# Paramset:
- lr scheduler

# To decide:
- net() vs net$forward()
- Wie machen wir es mit den weights? optimizer vs class weights in der loss function?

# Dataloader
--> we should maybe only make one dataset and then use dataset_subset (from mlverse::torch)
- implement set / row_ids when the test_ids PR is done

# Tests:
  - always test the edge case with batch_size = 1 (keepdim = TRUE)

# Visualization:

  - Plot tensor dimensions
  - Also plot channels for something like attention (otherwise unclear what key and value is)


Nächste Schritte:
- Ziel: Präsentation
- Bis dahin:
    - Branching fertig
    - Lukas und meine Arbeit integrieren
    - Man kann eigene Architektur angeben
- Vignette: 3 Levels: TorchOp, Preconfigured, Komplett flexibel
- Vereinigen: make_dataloder für verschiedene Datentypen + Füge preprocessing options hinzu
Bis: 3.5.
- TorchOpAttention (mit key, value input) --> test dass auch mehrere input channels von
verschiedenen nodes kein problem sind. Evtl auch mit mehrerern outputs (?)
- Define a type for the info we sent along the torchops


Steps for the graph architecture:
- How do I deal with the input definitions of the "mergers"
- FastAI: FitoneBatch FitOneEpoch FitAll fastai/learner.py


Long term goals
- Do decent logging
- be able to easily reparametrize the graph: something like the code below should produce a graph
with the parameter x that is set as a.out_features and b.out_features
- Overwrite %>>% to automatically inPipe
crement the ids (e.g. for relu)
- Be able to plot the network (with the tensor dimensions at each node)

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
