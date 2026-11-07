import numpy as np
import pickle
import multiprocessing as mp
import os
import toml
import time
from tqdm import tqdm

from em_mlgssm import EMmlgssm, InitEMmlgssm

NAME = 'clustering_and_parameter_estimation'
config = toml.load('config.toml')[NAME]

# Variables loaded directly from config.toml
DATA_DIR = config['DATA_DIR']
NUM_CLUSTER = config['NUM_CLUSTER']
DIM_X = config['DIM_X']
DIM_Y = config['DIM_Y']
NUM_DATA = config['NUM_DATA']
LENGTH = config['LENGTH']
NUM_CPU = config['NUM_CPU']
NUM_LGSSM = config['NUM_LGSSM']
MAX_ITER = config['MAX_ITER']
EPSILON = config['EPSILON']
FIX = config['FIX']
NUM_PERFORM = config['NUM_PERFORM']


if __name__ == "__main__": 

    # Ensure results directory exists
    os.makedirs("result", exist_ok=True)

    dataset = np.load(DATA_DIR).reshape(NUM_DATA, LENGTH, DIM_Y, 1)
    
    for m in NUM_LGSSM:
        for n in NUM_PERFORM:

            # Initialize InitEMmlgssm using the config parameters
            init_em = InitEMmlgssm(
                n_clusters=NUM_CLUSTER, 
                dim_x=DIM_X, 
                dim_y=DIM_Y, 
                n_cpu=NUM_CPU
            )

            # Assigning necessary variables for the decoupled tuning extraction phase
            init_em.set_y = dataset
            init_em.u_x, init_em.u_y = None, None
            
            # AS REQUESTED: Explicitly following data in config.toml rather than library defaults
            init_em.iter = MAX_ITER
            init_em.epsilon = EPSILON
            init_em.fix = FIX
            init_em.m = m
            
            # METHODOLOGY ALIGNMENT: Keeping seed as None to mirror original code behavior.
            init_em.seed = None

            print(f"--- Running Local Parameter Extraction Once for Config M={m}, Repetition={n} ---")
            
            total_tasks = len(init_em.set_y)
            
            # Safety range boundary mapping to match internal library design dynamically
            with mp.Pool(NUM_CPU) as pool:
                async_result = pool.map_async(init_em._tuning, range(total_tasks))
                
                # Monitor the async object updates cleanly via a tqdm tracking window
                with tqdm(total=total_tasks, desc="Extracting Parameters", unit="loc") as pbar:
                    while not async_result.ready():
                        # Unpack number of chunks left to compute processed tasks safely
                        try:
                            remaining = async_result._number_left
                            completed = total_tasks - remaining
                            if completed > pbar.n:
                                pbar.update(completed - pbar.n)
                        except AttributeError:
                            pass
                        time.sleep(0.5)
                    
                    # Force progress bar to fill completely once execution resolves
                    if pbar.n < total_tasks:
                        pbar.update(total_tasks - pbar.n)
                
                params_vec = np.asarray(async_result.get())

            # =====================================================================
            # NEW CONVENTIONS & STANDARDIZATION IMPLEMENTATIONS
            # =====================================================================
            # Save raw optimal matrix array back for Orchestrator structural reading
            np.save("result/lgssm_optimal_params.npy", params_vec)

            # Change 1: Save baseline final parameters using 'optimal' instead of 'best'
            csv_path = f"result/optimal_parameters_m{m}_n{n}.csv"
            np.savetxt(
                csv_path, 
                params_vec, 
                delimiter=",", 
                header="mu,P,A,Gamma,C,Sigma", 
                comments="",
                fmt='%.18f'
            )
            print(f"Saved self-documenting parameters to: {csv_path}")

            # Change 2: Global Standardization (Single Mean and Variance across the entire data matrix)
            global_mean = np.mean(params_vec)
            global_std = np.std(params_vec)
            global_std_safe = global_std if global_std > 0 else 1.0
            params_global_std = (params_vec - global_mean) / global_std_safe
            
            np.savetxt(
                "result/global-std-lgssm-optimal-paras.csv", 
                params_global_std, 
                delimiter=",", 
                header="mu,P,A,Gamma,C,Sigma", 
                comments="",
                fmt='%.18f'
            )

            # Change 3: Local Standardization (Individual Mean and StDev for each separate parameter attribute column)
            local_mean = np.mean(params_vec, axis=0)
            local_std = np.std(params_vec, axis=0)
            local_std_safe = np.where(local_std == 0, 1.0, local_std)
            params_local_std = (params_vec - local_mean) / local_std_safe
            
            np.savetxt(
                "result/local-std-lgssm-optimal-paras.csv", 
                params_local_std, 
                delimiter=",", 
                header="mu,P,A,Gamma,C,Sigma", 
                comments="",
                fmt='%.18f'
            )
            # =====================================================================

            # Vary K clusters sequentially without retraining the baseline parameters
            VARYING_K = [2,3,4,5,6,7,8] 
            
            for k_val in VARYING_K:
                print(f"  -> Running Clustering and MLGSSM for K = {k_val}")
                
                # Update the target cluster quantity profile dynamically 
                init_em.M = k_val 
                
                # Execute K-means division mapping on the static parameter matrix
                centers_vec, weights = init_em._kmeans(params_vec)
                init_params = init_em._vec_to_array(centers_vec)
                init_params['weight'] = weights

                # Build the global mixture state engine
                model = EMmlgssm(
                    state_mats=init_params["A"], 
                    state_covs=init_params["Gamma"], 
                    obs_mats=init_params["C"], 
                    obs_covs=init_params["Sigma"], 
                    init_state_means=init_params["mu"], 
                    init_state_covs=init_params["P"], 
                    weights=init_params["weight"]
                )

                # Running the macro MLGSSM framework with structural configurations from config.toml
                result = model.fit(
                    Y=dataset, 
                    max_iter=MAX_ITER, 
                    epsilon=EPSILON, 
                    n_cpu=NUM_CPU, 
                    fix_param=FIX,
                    bic=False
                )

                # Save output pickles partitioned distinctly by cluster variation tags
                with open(f"result/parameter_{m}_{n}_k{k_val}.pickle", mode="wb") as f:
                    pickle.dump(result['parameter'], f)

                with open(f"result/label_{m}_{n}_k{k_val}.pickle", mode="wb") as f:
                    pickle.dump(result['label'], f)