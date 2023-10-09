* merging of graphs now uses hashes instead of `identical()`
* In order to enable the preprocessing of lazy tensors we always clone the graph
  and then modify a cloned graph.
  Doing this ensures that


Why other approaches don't work
* One idea was to communicate to the pipeop whether they are an input pipeop and clone the preprocessing
  graph only in this case.
  This is tricky with things like PipeOpNOP and we would risk modifying user-input in place.
  The only solution is to properly do the cloning of the preprocessing graphs
  and then afterwards merge them.
  This means that in a graph that would look ass follows in a ModelDescriptor:
          --> C
  A --> B
          --> D

  is turned into

  A1 --> B1 --> C

  and

  A2 --> B2 --> D

  When we then want to merge A1 with A2 they will no longer be identical but their hash should be the same.



