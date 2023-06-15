import numpy as np


def chunkit(str, n):
    return [str[i:i+n] for i in range(0, len(str), n)]


def get_max_num(speclab):
    specsyms = 'spdfghiklmnoqrtuv'
    l = specsyms.index(speclab[0])
    plus_els = 0 if speclab[1].strip() else 2
    return 2 * l + plus_els


def readj2(lab_str, defj2):
    if lab_str.strip()=='':
        return defj2
    strspl = lab_str.split('/')
    j2 = int(strspl[0])
    if len(strspl)==1:
        j2 *= 2
    return j2


def lines_to_csf(orbs, csf_3lines, J2tot):

    cells = [0] * (3*len(orbs))
    
    num_line = csf_3lines[0].rstrip()
    J2nu_line = csf_3lines[1].rstrip().ljust(len(num_line))
    J2cpl_line = csf_3lines[2].rstrip()[5:-4].ljust(len(num_line))

    num_chunk = chunkit(num_line, 9)
    J2nu_chunk = chunkit(J2nu_line, 9)
    J2cpl_chunk = chunkit(J2cpl_line, 9)

    lenchunks = len(num_chunk)
    J2cpl_lst = 0
    for ich in range(0, lenchunks):
        orb = num_chunk[ich][:5]
        num = int(num_chunk[ich][6:8])
        maxnum = get_max_num(orb[3:5])
        J2nu = J2nu_chunk[ich].split(';')
        J2 = readj2(J2nu[-1], 0)
        nu = int(J2nu[0]) if len(J2nu)==2 else np.nan
        if J2cpl_lst == 0:
            J2cpl_lst = J2
            J2cpl = J2
        elif J2 == 0:
            J2cpl = J2cpl_lst
        else:
            J2cpl = readj2(J2cpl_chunk[ich], J2cpl_lst) if ich < lenchunks-1 else J2tot
            J2cpl_lst = J2cpl

        iorb = 3 * orbs.index(orb)
        cells[iorb : iorb+3] = num/maxnum, J2/J2tot, J2cpl/J2tot

    for ii, num in enumerate(cells[::3]):
        if (num == 0) and ii != 0:
            cellstart_this = ii * 3
            cellstart_last = (ii - 1) * 3
            Jcpl_lst = cells[cellstart_last + 2]
            if Jcpl_lst != 0:
                cells[cellstart_this + 2] = Jcpl_lst
                
    return cells


def line_to_orbs(line):
    orb_line = line.rstrip()
    orbs = chunkit(orb_line, 5)
    orbs[-1] = orbs[-1].ljust(5)
    return orbs


def extract_orbs(flnm_head):
    with open(flnm_head, "r") as f_head:
        for _ in range(0,3):
            f_head.readline()
        orbsline = f_head.readline()
        orbs = line_to_orbs(orbsline)
    return orbs


def count_prim_pool(flnm_full, flnm_head):

    with open(flnm_full, "r") as f_full:
        with open(flnm_head, "r") as f_head:
            for _ in range(0,5):
                f_head.readline()
                f_full.readline()

            csfs_prim_num = 0
            while True:
                ln = f_head.readline()
                if not ln:
                    break
                else:
                    f_head.readline()
                    f_head.readline()
                    f_full.readline()
                    f_full.readline()
                    f_full.readline()
                    csfs_prim_num += 1

        csfs_pool_num = 0
        while True:
            ln = f_full.readline()
            if not ln:
                break
            else:
                f_full.readline()
                f_full.readline()
                csfs_pool_num += 1

    return csfs_prim_num, csfs_pool_num


def produce_basis_npy(flnm_npy, flnm_full, flnm_head, J2tot):

    csfs_prim_num, csfs_pool_num = count_prim_pool(flnm_full, flnm_head)
    orbs = extract_orbs(flnm_head)
    
    csfs_np = np.zeros( ( csfs_pool_num, 3*len(orbs) ), dtype=np.float32)
    
    with open(flnm_full, "r") as f_full:
        for _ in range(5 + 3*csfs_prim_num):
            f_full.readline()

        for csf_ii in range(csfs_pool_num):
            ln1 = f_full.readline()
            ln2 = f_full.readline()
            ln3 = f_full.readline()
            csfs_np[csf_ii, :] = lines_to_csf(orbs, [ln1, ln2, ln3], J2tot)

    with open(flnm_npy, "wb") as f_npy:
        np.save(f_npy, csfs_np)
            
    return None
