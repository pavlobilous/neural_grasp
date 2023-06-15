from process_grasp_inp import *

filenames = {
    "full_basis_npy": "rcsf.npy",
    "full_grasp_inp": "rcsf.full",
    "head_grasp_inp": "rcsf.head"
}

J2tot = 5

produce_basis_npy(
    filenames["full_basis_npy"],
    filenames["full_grasp_inp"],
    filenames["head_grasp_inp"],
    J2tot
)

