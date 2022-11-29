import numpy as np
import ase
import matplotlib.pyplot as plt
from ase.io import read, write

# this needs github.com/lab-cosmo/librascal, commit ID 3fd08e4bdb70060ad797ffc63fb591858bb62661; 
# https://github.com/lab-cosmo/librascal/releases/tag/cnn-counterexample
from rascal.representations import SphericalExpansion, SphericalInvariants
from rascal.utils import (get_radial_basis_covariance, get_radial_basis_pca, 
                          get_radial_basis_projections, get_optimal_radial_basis_hypers )
from rascal.utils import radial_basis
from rascal.utils import WignerDReal, ClebschGordanReal, spherical_expansion_reshape, lm_slice, real2complex_matrix, compute_lambda_soap
from rascal.models import gaptools, KRR
import argparse

# builds learning curves for the n-mers dataset and for the predictions on the selected structures file.
# outputs results in a dictionary that can be further processed to make figures & diagnostics


parser = argparse.ArgumentParser(description='Computes gap lc')
parser.add_argument('seed', type=int,
                    help='random seed')


args = parser.parse_args()

seed = args.seed

spherical_expansion_hypers = {
    "interaction_cutoff": 5,
    "max_radial": 9,
    "max_angular": 7,
    "gaussian_sigma_constant": 0.25,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.5,
    "radial_basis": "GTO",
    "cutoff_function_type" :"RadialScaling",
    "cutoff_function_parameters" :
                    dict(
                            rate=2,
                            scale=3.5,
                            exponent=2
                        ),
}

spex = SphericalExpansion(**spherical_expansion_hypers)
mycg = ClebschGordanReal(spherical_expansion_hypers["max_angular"])

selected = read("data/selected-structures.xyz", ":")
y_ref = []
for i, f in enumerate(selected):
    f.cell = [100,100,100]
    f.positions += 50
    y_ref.append((f.info["energy_ha"]-selected[0].info["energy_ha"]*len(f.numbers)/len(selected[0].numbers))* 27.211386)

frames = read("data/mbpol-dimer-b3lyp_ccpvtz-fps.xyz", ":")
frames += read("data/mbpol-trimer-b3lyp_ccpvtz-fps.xyz", ":")
frames += read("data/tetramers-b3lyp_ccpvtz-fps.xyz", ":")
y_full = np.zeros(len(frames))
f_full = []
max = 0
for i, f in enumerate(frames):
    f.cell = [100,100,100]
    f.positions += 50
    y_full[i] = f.info["energy_b3lyp_ha"] - selected[0].info["energy_ha"]*len(f.numbers)/len(selected[0].numbers)
    y_full[i] *= 27.211386 # to eV
    f_full.append(-f.arrays["gradient_b3lyp_habohr"]*27.211386/0.529177) # converts to eV/Angstrom


    
# removes mean so atomic energy is zero
at_energy = {1: 0.0, 8:0.0}
for item in f_full:
    print(np.shape(item))

print(max)

spherical_expansion_hypers = get_optimal_radial_basis_hypers(spherical_expansion_hypers, frames[::2], expanded_max_radial=16)

spherical_expansion_hypers["soap_type"] = "PowerSpectrum"
spherical_expansion_hypers["normalize"] = True
soap = SphericalInvariants(**spherical_expansion_hypers)

soap, feature_list = gaptools.calculate_features(frames, spherical_expansion_hypers, auto_wrap=True)
n_sparse = {1: 5000, 8: 5000}
X_sparse = gaptools.sparsify_environments(soap, feature_list, n_sparse, selection_type="FPS")

do_gradients=False
np.random.seed(seed)
itrain = np.arange(len(frames))
np.random.shuffle(itrain)

hypers_grad = spherical_expansion_hypers.copy()
hypers_grad['compute_gradients'] = do_gradients
soap_grad = SphericalInvariants(**hypers_grad)

lc = {}
lc["ref"] = y_ref
lc["itrain"] = itrain
lc["seed"] = seed
lc["n_sparse"] = n_sparse
lc["hypers"] = spherical_expansion_hypers
lc["gradients"] = do_gradients
for ntrain in [100, 200, 500, 1000, 2000, 5000, 10000, 15000]:
    print(f"Running LC for {ntrain} structures")
    train_frames = [frames[f] for f in itrain[:ntrain]]
    y_train = y_full[itrain[:ntrain]]
    f_train = np.vstack([f_full[f] for f in itrain[:ntrain]])


    soap, feature_list = gaptools.calculate_features(train_frames, spherical_expansion_hypers, auto_wrap=True)    
    if do_gradients:
        (k_obj, K_sparse_sparse, K_full_sparse, K_grad_full_sparse) = gaptools.compute_kernels(
            soap,
            feature_list,
            X_sparse,
            do_gradients = do_gradients,
            soap_power=4
        )
    else:
            (k_obj, K_sparse_sparse, K_full_sparse) = gaptools.compute_kernels(
            soap,
            feature_list,
            X_sparse,
            do_gradients = do_gradients,
            soap_power=4
        )
    
    best_rmse = np.inf
    best_alpha = 0
    cv = {}
    for alpha in [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        weights = gaptools.fit_gap_simple(
            train_frames,
            K_sparse_sparse,
            y_train,
            K_full_sparse,
            energy_regularizer_peratom=alpha,
            forces=f_train if do_gradients else None,
            kernel_gradients_sparse=K_grad_full_sparse if do_gradients else None,
            energy_atom_contributions=at_energy,
            force_regularizer=1E-2 if do_gradients else None,
            jitter = 1e-7,
            solver="RKHS"    
        )
        model = KRR(weights, k_obj, X_sparse, at_energy,
            description="GAP for the dimers/trimers/tetramers dataset")                

        # make predictions on the test set
        y_pred = []
        f_pred = []

        itest = itrain[-2000:]
        for f in itest:
            m = soap_grad.transform(frames[f])
            y_pred.append(model.predict(m))
            f_pred.append(model.predict_forces(m))

        y_pred = np.array(y_pred).flatten()
        f_pred = np.vstack(f_pred)
        
        mae = np.abs(y_pred - y_full[itest]).mean()        
        rmse = np.sqrt( ((y_pred - y_full[itest])**2).mean() )
        
        cv[alpha] = (mae, rmse)
        
        print(f"CV stats for {alpha}: RMSE: {rmse}, MAE: {mae}")
        if rmse<best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_pred = y_pred
            best_fpred = f_pred
            
    weights = gaptools.fit_gap_simple(
            train_frames,
            K_sparse_sparse,
            y_train,
            K_full_sparse,
            energy_regularizer_peratom=best_alpha,
            forces=f_train if do_gradients else None,
            kernel_gradients_sparse=K_grad_full_sparse if do_gradients else None,
            energy_atom_contributions=at_energy,
            force_regularizer=1E-2 if do_gradients else None,
            jitter = 1e-7,
            solver="RKHS"    
        )
        
    model = KRR(weights, k_obj, X_sparse, at_energy,
        description="GAP for the dimers/trimers/tetramers dataset")           

    # make predictions on the test set
    y_pred = []
    f_pred = []

    itest = itrain[-2000:]
    for f in itest:
        m = soap_grad.transform(frames[f])
        y_pred.append(model.predict(m))
        f_pred.append(model.predict_forces(m))

    y_pred = np.array(y_pred).flatten()
    f_pred = np.vstack(f_pred)
    
    mae = np.abs(y_pred - y_full[itest]).mean()        
    rmse = np.sqrt( ((y_pred - y_full[itest])**2).mean() )
        
    # make predictions on the test set
    y_counter = []
    f_counter = []
    y_ref = []
    for f in selected:     
        y_ref.append((f.info["energy_ha"]-selected[0].info["energy_ha"]*len(f.numbers)/len(selected[0].numbers))* 27.211386)
        m = soap_grad.transform(f)
        y_counter.append(model.predict(m))
        f_counter.append(model.predict_forces(m))

    y_counter = np.array(y_counter).flatten()
    f_counter = np.vstack(f_counter)
    print("Signed errors on selected")
    print(y_counter - y_ref)
    
    lc[ntrain] = (y_counter, y_pred, cv)
    
    np.save(f"gap_learning_curves/lc-{seed}.npy", lc)
    
