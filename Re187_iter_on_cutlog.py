import tensorflow as tf
from neural_grasp import *



filenames = {
    "full_basis_npy": "rcsf.npy",
    "full_grasp_inp": "rcsf.full",
    "head_grasp_inp": "rcsf.head",
    "curr_grasp_inp": "rcsf.inp",
    "radial_wf": "rwfn.prim.inp",
    "grasp_out_weights": "csfwgt_rmcdhf_mpi.dat",
    "rmcdhf_sum": "rmcdhf.sum",
    "grasp_script": "run_grasp.sh"
}

ngm = NeuralGraspManager(filenames)



es = tf.keras.callbacks.EarlyStopping( monitor='val_accuracy', patience=3, restore_best_weights=True )
fit_params = { 
    'epochs': 200,
    'verbose': 2,
    'validation_split': 0.2,          
    'callbacks': [es]
}        

ngm.load_state( "save_init_random", fit_params )



cutlog = -8.6
balancing_ratio = 1.0

ngm.prepare_next_atcomp_nn( cutlog, balancing_ratio, appl_max_at_once=250000 )

ngm.save_state( folder=f"save_cutlog_{cutlog:.1f}" )


ngm.print_logs()
