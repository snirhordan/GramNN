Reference n-mers structures and energetics
==========================================

The folder contains ASE extended xyz formatted files containing 
geometries and B3LYP-CCPVTZ energies and forces for different 
types of small water clusters

mbpol-dimer-b3lyp_ccpvtz-fps.xyz
mbpol-trimer-b3lyp_ccpvtz-fps.xyz  

Farthest-point-sampling subselection of structures from the MBPol
training data [DOI: 10.1021/ct5004115]

tetramers-b3lyp_ccpvtz-fps.xyz

Farthest-point-sampling subselection of a replica exchange simulation
of the water tetramer

selected-structures.xyz

Selected structures for testing and validation:
frame 0: geometry-optimized water monomer, for setting the cohesive energy zero
frame 1: geometry-optimized ground state water tetramer
frames 2,3:   A_1^+- degenerate pair
frames 4,5:   A_2^+- degenerate pair


nmers.db
Concatenation of dimers, trimers, tetramers, converted to schnetpack formaty

