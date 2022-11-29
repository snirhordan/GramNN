These scripts allow to reproduce the numerical experiments discussed in
"Incompleteness of graph convolutional neural networks for points clouds in three dimensions"

The data/ folder contains structures that are used for training and testing. 

wltest.py   contains simple functions to perform a Weisfeiler-Lehman tests on a distance-decorated molecular graph 
gap-lc.py   builds a SOAP-GAP model of the energy (and optionally forces) of water n-mers and tests it on representative
            degenerate pair structures that are tetramer-like
spk_eval.py after training a SchNet model with the standard tools in the schnetpack libraries, this script simply 
            computes energies for the degenerate pairs. a rather trivial post-processing


Furthermore, we include additional examples of structures that defy GNNs in the examples/ folder 
(specifically some ethene conformers, and bulk carbon structures) as well as a Jupyter notebook
`2_body_gnn_can_compute_angles.ipynb` that demonstrates how a dGNN (SchNET) is capable of predicting
angular information for configurations composed of a single triangle. 
