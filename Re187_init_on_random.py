from neural_grasp import *
import tensorflow as tf


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




def get_new_model(num_params):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(num_params//3, 3)))
    model.add(tf.keras.layers.Conv1D(96, 3, activation="relu"))
    model.add(tf.keras.layers.Conv1D(16, 1, activation='relu'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(150, activation='relu'))
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(90, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    return model


ngm.initialize_state(get_new_model, fit_params=None)


ngm.prepare_first_atcomp_rand( random_frac=0.01 )
ngm.save_state( folder="save_init_random" )


ngm.print_logs()
