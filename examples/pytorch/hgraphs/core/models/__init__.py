"""
Contains the definitions for the Encoder and Decoder, their combination
as Autoencoder, a NodePredictor to predict motifs and attachment configs,
an AttachmentPredictor to predict the pair of atoms at which 2 motifs
will attach, and a utility to easily embed parts of the hgraph, embed().

Abbreviations:
hgraph := hierarchical graph
attch := attachment
pred := predicted
ctx := context
idx := index
"""
