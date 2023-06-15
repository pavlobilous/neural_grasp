import time
import io
import os
import numpy as np


class NeuralGraspManager:

    def print_logs(self):
        print(self.logbuffer.getvalue())
        return None


    def load_basis_npy(self):
        self.basis_params = np.load(self.filenames["full_basis_npy"], mmap_mode='r')
        self.basis_size, self.params_num = self.basis_params.shape
        return None


    def read_prim_num(self):
        with open(self.filenames["head_grasp_inp"], "r") as f_head:
            while True:
                ln = f_head.readline()
                if "CSF" in ln:
                    break
            prim_cnt = 0
            while True:
                ln = f_head.readline()
                if not ln:
                    break
                else:
                    ln = f_head.readline()
                    ln = f_head.readline()
                    prim_cnt += 1
        self.prim_num = prim_cnt
        return None
    
    
    def __init__(self, filenames):
        self.filenames=filenames
        self.load_basis_npy()
        self.read_prim_num()
        
        self.logbuffer = io.StringIO()
        self.logbuffer.write("\n**************************\n")
        self.logbuffer.write(f"Neural GRASP manager created.\n")
        self.logbuffer.write(f"Full basis contains {self.basis_size} non-primary states\n")
        self.logbuffer.write(f"\tcharacterized by {self.params_num} parameters.\n")
        self.logbuffer.write(f"Number of primary states: {self.prim_num}\n")
        return None


    def initialize_arrays(self):
        self.onoff = np.zeros(self.basis_size, dtype=bool)
        self.wgts = np.zeros(self.basis_size, dtype=np.float128)
        self.logbuffer.write("Arrays initialized.\n")
        return None


    def initialize_nnsetup(self, get_new_model, fit_params):
        self.model = get_new_model(self.params_num)
        self.fit_params = fit_params
        self.logbuffer.write("NN-setup initialized.\n")
        return None

    
    def initialize_state(self, get_new_model, fit_params):
        self.logbuffer.write("Initialization...\n")
        self.initialize_arrays()
        self.initialize_nnsetup(get_new_model, fit_params)
        return None

    
    def save_arrays(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
        
        arrays_to_save = {}
        arrays_to_save["onoff"] = self.onoff
        arrays_to_save["wgts"] = self.wgts
        arrays_to_save["mark_train"] = self.mark_train
        arrays_to_save["mark_apply"] = self.mark_apply
        
        for arr_name, arr_npy in arrays_to_save.items():
            arr_path = os.path.join(folder, f"{arr_name}.npy")
            with open(arr_path, "bw") as f:
                np.save(f, arr_npy)

        if os.path.exists(self.filenames["grasp_out_weights"]):
            os.system(f'cp {self.filenames["grasp_out_weights"]} {folder}')

        if os.path.exists(self.filenames["rmcdhf_sum"]):
            os.system(f'cp {self.filenames["rmcdhf_sum"]} {folder}')
            
        self.logbuffer.write("Arrays saved.\n")
        return None

    
    def save_nnsetup(self, folder):
        self.model.save(folder)
        self.logbuffer.write("NN-setup saved (model only).\n")
        return None

    
    def save_state(self, folder):
        self.logbuffer.write("Saving...\n")
        arrays_folder = os.path.join(folder, "arrays")
        nnsetup_folder = os.path.join(folder, "model")        
        self.save_arrays(arrays_folder)
        self.save_nnsetup(nnsetup_folder)
        return None

    
    def load_arrays(self, folder):
        arrays_loaded = {}
        for arr_name in "onoff wgts mark_train mark_apply".split():
            arr_path = os.path.join(folder, f"{arr_name}.npy")
            with open(arr_path, "rb") as f:
                arrays_loaded[arr_name] = np.load(f)

        self.onoff = arrays_loaded["onoff"]
        self.wgts = arrays_loaded["wgts"]
        self.mark_train = arrays_loaded["mark_train"]
        self.mark_apply = arrays_loaded["mark_apply"]
                                       
        self.logbuffer.write("Arrays loaded.\n")
        return None

    
    def load_nnsetup(self, folder, fit_params):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(folder)
        self.fit_params = fit_params
        self.logbuffer.write("NN-setup loaded.\n")
        return None

    
    def load_state(self, folder, fit_params):
        self.logbuffer.write("Loading...\n")
        arrays_folder = os.path.join(folder, "arrays")
        self.load_arrays(arrays_folder)
        nnsetup_folder = os.path.join(folder, "model")
        self.load_nnsetup(nnsetup_folder, fit_params)
        return None


    def write_atcomp_input(self):
        with open(self.filenames["curr_grasp_inp"], "w") as f_curr:
            with open(self.filenames["full_grasp_inp"], "r") as f_full:
                with open(self.filenames["head_grasp_inp"], "r") as f_head:
                    while True:
                        ln = f_head.readline()
                        if not ln:
                            break
                        else:
                            f_full.readline()
                            f_curr.write(ln)
                for csfs_ind in range(self.basis_size):
                    ln1 = f_full.readline()
                    ln2 = f_full.readline()
                    ln3 = f_full.readline()
                    if self.onoff[csfs_ind]:
                        f_curr.write(ln1)
                        f_curr.write(ln2)
                        f_curr.write(ln3)                
        os.system('cp ' + self.filenames["radial_wf"] + ' rwfn.inp')
        return None


    def read_atcomp_output(self):
        
        with open(self.filenames["grasp_out_weights"], 'r') as f:
            csfwgts = f.readlines()

        for i in range(self.prim_num, len(csfwgts)):
            strs = csfwgts[i].strip().split('E')
            if len(strs)==1:
                strs.append('0')
            csfwgts[i] = ( float(strs[0]) * 10**int(strs[1]) )**2

        self.wgts[self.onoff] = csfwgts[self.prim_num:]
        return None


    def prepare_first_atcomp_rand(self, random_frac):
        self.logbuffer.write("\n**************************\n")
        self.logbuffer.write("Preparation for first atomic computation on random states...\n")
        
        init_num = int(self.basis_size * random_frac)
        self.logbuffer.write(f"Chosen random fraction: {random_frac}\n")
        self.logbuffer.write(f"\tresulting in {init_num} non-primary initial basis states.\n")
        
        part_ind = np.random.choice(self.basis_size, init_num, replace=False)
        self.onoff[part_ind] = True
        self.mark_train = self.onoff.copy()
        self.mark_apply = ~self.onoff
        
        self.write_atcomp_input()
        self.logbuffer.write("Ready for the atomic computation.\n")
        return None


    def adjust_features_to_nn(self, X):
        nn_inp_shape = self.model.input_shape
        if len(nn_inp_shape) > 2:
            nn_channels_shape = nn_inp_shape[2:]
            return X.reshape(X.shape[0], -1, *nn_channels_shape)
        else:
            return X

        
    def adjust_answers_to_nn(self, y):
        import tensorflow as tf
        return tf.keras.utils.to_categorical(y)

    
    def nn_prediction_to_importance(self, y):
        return (y[:, 1] >= 0.5)
    
    
    def nn_train_on_cutlog(self, cutlog):

        import tensorflow as tf

        self.logbuffer.write(f"NN-training on cutlog: {cutlog}\n")
        cutoff = 10**cutlog

        X_train = self.basis_params[self.mark_train, :]
        y_train = (self.wgts[self.mark_train] > cutoff)
        
        X_train = self.adjust_features_to_nn(X_train)
        y_train = self.adjust_answers_to_nn(y_train)

        iperm = np.random.permutation(X_train.shape[0])
        X_train = X_train[iperm]
        y_train = y_train[iperm]
        
        time_nntrain_start = time.perf_counter()
        hist = self.model.fit(X_train, y_train, **self.fit_params)
        time_nntrain_finish = time.perf_counter()
        duration_nntrain = time_nntrain_finish - time_nntrain_start
        rep_xtrain_size = X_train.shape[0]
        rep_ytrain_sum = int(y_train[:, 1].sum())

        gpus = tf.config.list_physical_devices('GPU')
        train_device = "GPU" if len(gpus) > 0 else "CPU"
        
        self.logbuffer.write(f"NN trained on: {rep_xtrain_size}\n")
        self.logbuffer.write(f"\tof which important: {rep_ytrain_sum};\n")
        self.logbuffer.write(f"\ttraining time: {duration_nntrain:.2f} sec\n")
        self.logbuffer.write(f"\ton device: {train_device}.\n")
        self.logbuffer.write("NN training history:\n")
        val_accs = hist.history["val_accuracy"]
        for i, va in enumerate(val_accs):
            self.logbuffer.write(f"\tEpoch {i}: val_accuracy: {va}\n")

        return None
    
            
    def nn_apply_to_pool(self, max_at_once_inp=None):
        max_at_once = max_at_once_inp or self.basis_size
        
        duration_nnapply = 0
        num_nnapply = 0
        for ichunk, lw_ind in enumerate( range(0, self.basis_size, max_at_once) ):
            up_ind = min(lw_ind + max_at_once, self.basis_size)
            X_apply = self.adjust_features_to_nn(self.basis_params[lw_ind : up_ind, :])
            time_nnapply_start = time.perf_counter()
            y_apply_new = self.model(X_apply).numpy()
            y_apply = y_apply_new if ichunk == 0 else np.vstack([y_apply, y_apply_new])
            time_nnapply_finish = time.perf_counter()
            duration_nnapply_chunk = time_nnapply_finish - time_nnapply_start
            #print(f"------> NN application: chunk {ichunk} : {duration_nnapply_chunk:.2f} sec.")
            duration_nnapply += duration_nnapply_chunk
            num_nnapply += X_apply.shape[0]
            
        y_apply = y_apply[self.mark_apply, :]
        predicted_impt = self.nn_prediction_to_importance(y_apply)
        predicted_impt_inds = np.where(self.mark_apply)[0][predicted_impt]
        predicted_notimpt_inds = np.where(self.mark_apply)[0][~predicted_impt]
        rep_predimpt_sum = predicted_impt.sum()

        self.logbuffer.write(f"NN applied to: {num_nnapply};\n")
        if max_at_once_inp:
            self.logbuffer.write(f"\tlimited to {max_at_once} states at once;\n")
        self.logbuffer.write(f"\tprediced to be important: {rep_predimpt_sum};\n")
        self.logbuffer.write(f"\tapplication time: {duration_nnapply:.2f} sec.\n")

        return predicted_impt_inds, predicted_notimpt_inds

            
    def prepare_next_atcomp_nn(self, cutlog, balancing_ratio, appl_max_at_once=None):

        self.logbuffer.write("\n**************************\n")
        self.logbuffer.write("Preparation for next atomic computaion using NN...\n")

        self.read_atcomp_output()
        self.nn_train_on_cutlog(cutlog)

        predicted_impt_inds, predicted_notimpt_inds = self.nn_apply_to_pool(appl_max_at_once)

        cutoff = 10**cutlog
        self.onoff[self.mark_train & (self.wgts < cutoff)] = False
        self.mark_train[:] = False
        self.mark_train[predicted_impt_inds] = True
        self.mark_apply[:] = ~self.onoff
        self.mark_apply[predicted_impt_inds] = False

        if balancing_ratio:
            num_bal_req = int(balancing_ratio * predicted_impt_inds.shape[0])
            num_bal_max = predicted_notimpt_inds.shape[0]
            if num_bal_req < num_bal_max:
                bal_inds = np.random.choice(predicted_notimpt_inds, num_bal_req, replace=False)
            else:
                raise RuntimeError("Pool exhausted when balancing.")
            self.mark_train[bal_inds] = True
            self.mark_apply[bal_inds] = False
            
            self.logbuffer.write(f"Chosen balancing ratio: {balancing_ratio}\n")
            self.logbuffer.write("\t[defined as (num. of extra random states)/(num. of NN-selected states)]\n")
                        
        self.onoff[self.mark_train] = True
        
        self.write_atcomp_input()

        self.logbuffer.write(f"Marked for NN training in next iteration: {self.mark_train.sum()}\n")
        self.logbuffer.write(f"Marked for NN application in next iteration: {self.mark_apply.sum()}\n")
        self.logbuffer.write(f"Prepared in total {self.onoff.sum()} non-primary states.\n")
        self.logbuffer.write("Ready for the atomic computation.\n")
                       
        return None


    def final_trim_wf(self, cutlog):
        self.logbuffer.write("\n**************************\n")
        self.logbuffer.write("Final WF trimming...\n")
        self.logbuffer.write(f"Chosen cutlog: {cutlog}\n")

        self.read_atcomp_output()

        cutoff = 10**cutlog
        self.onoff[self.mark_train & (self.wgts < cutoff)] = False

        self.write_atcomp_input()
        
        self.logbuffer.write(f"Obtained in total {self.onoff.sum()} non-primary states.\n")
        self.logbuffer.write("Computation finished.\n")
        self.logbuffer.write("\n**************************\n")
        
        return None


    def launch_atcomp(self):
        self.logbuffer.write(f"Running atomic computation...")

        time_atcomp_start = time.perf_counter()
        os.system(f'sh {self.filenames["grasp_script"]}')
        time_atcomp_finish = time.perf_counter()
        duration_atcomp = time_atcomp_finish - time_atcomp_start

        with open(self.filenames["rmcdhf_sum"], 'r') as f:
            while not f.readline().startswith('Eigenenergies:'):
                pass
            f.readline()
            f.readline()
            f.readline()
            ln = f.readline()

            enstr = ln.split()[-1].split('D')
        ennum = float(enstr[0][1:])
        enpow = int(enstr[1][1:])
        energy = ennum * 10**enpow

        self.logbuffer.write(f"Atomic computation finished.")
        self.logbuffer.write(f"\tenergy: {energy}\n")
        self.logbuffer.write(f"\tcomputation time: {duration_atcomp:.2f} sec\n")

        return energy
