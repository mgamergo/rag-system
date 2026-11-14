# RAG notes

Retrieval-augmented generation (RAG) has two stages: retrieval and generation.
Retrieval finds document chunks relevant to a question. Generation gives those
chunks and the question to a language model to produce an answer grounded in
the retrieved evidence.

Before retrieval, ingestion splits documents into overlapping chunks and turns
each chunk into an embedding. An embedding is a numeric vector. At query time,
the same embedding model encodes the question. Cosine similarity ranks the
stored chunk vectors against the question vector.

The generation model is separate from the embedding model. It reads the selected
chunks as context and cites the sources that support its answer. When the
context cannot answer the question, it should say that it does not know.
