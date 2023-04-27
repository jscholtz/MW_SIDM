import numpy as np
import h5py
from tqdm.notebook import tqdm
import scipy.fft as fft
import pathlib


def predict_mem(nside: int, slices: int) -> float:
    """Gives an estimate of memory use for the FFT process"""
    return 64*(nside/1024)**3/slices


def det_slices(nside: int, memory=4.0) -> np.array:
    """Tries to guess the right number of slices in order not to surpass the mem limit [GB]"""
    nsides = np.array(nside)
    pre = np.log((64/memory*(nsides/1024)**3))/np.log(2)
    mask = pre < 0
    pre[mask] = 0
    return np.power(2, (np.ceil(pre))).astype('i')


def det_slices_v1(nside: int, memory=4) -> int:
    """Tries to guess the right number of slices in order not to surpass the mem limit [GB]"""
    pre = np.log((64/memory*(nside/1024)**3))/np.log(2)
    if pre < 0:
        pre = 0
    return int(np.power(2, (np.ceil(pre))))


def time_estimate(nside: int) -> float:
    """Times estimate for the FFT process"""
    a0 = 3.292
    b0 = -17.683
    return nside**a0*np.exp(b0)


def generate_props(nside: int, boundary: float, slices=-1, fiducial=0.7, G_Newton=4.5e-6) -> dict:
    """Generate a dictionary that carries all the information about a given file/process"""
    mid = nside//2
    
    fmid = int(mid*fiducial)
    fid1 = mid-fmid
    fid2 = mid+fmid

    xs = np.linspace(-boundary, boundary, nside)
    dx = (xs[1]-xs[0])
    rs = np.sqrt(xs**2+2.0*(xs[mid])**2)
    ks = np.fft.fftfreq(nside, d=dx)
    rho0 = 1.0/dx**3
    
    if slices == -1:
        temp_slices = det_slices_v1(nside, memory=4)
    else:
        temp_slices = slices
    
    props = {'nside': nside, 'mid': mid, 'xs': xs, 'dx': dx, 'rs': rs, 'ks': ks, 'rho0': rho0, 'slices': temp_slices,
             'fmid': fmid, 'fid1': fid1, 'fid2': fid2, 'fidx': xs[fid1:fid2], 'fidr': rs[fid1:fid2], 'G': G_Newton}
    
    return props


def get_plane(filename: str, index: int, axis=0):
    with h5py.File(filename, 'r+') as F:
        dset = F['d1']
        if axis == 0:
            return dset[index, :, :]
        if axis == 1:
            return dset[:, index, :]
        if axis == 2:
            return dset[:, :, index]


def get_fullarray(filename: str) -> np.array:
    with h5py.File(filename, 'r') as F:
        if len(F['d1']) <= 512:
            return F['d1'][:]
        else:
            raise Exception(f"You are asking to directly handle too much data! [nside={len(F['d1'])}>512]")


def extract_fiducial(filename: str, fidname: str, props: dict):

    fid1 = props['fid1']
    fid2 = props['fid2']
    fmid = props['fmid']
    nside = props['nside']
    slices = props['slices']
    nn = nside//slices
    if nside < 512:
        with h5py.File(filename, 'r+') as F:
            with h5py.File(fidname, 'w') as Fid:
                dset = F['d1']
                Fid.create_dataset('d1', data=np.real(dset[fid1:fid2, fid1:fid2, fid1:fid2]))
    else:
        with h5py.File(filename, 'r+') as F:
            dset = F['d1']
            with h5py.File(fidname, 'w') as Fid:
                Fid.create_dataset('d0', (nside, 2*fmid, 2*fmid))
                temp = Fid['d0']
                for ii in tqdm(range(slices)):
                    temp[ii*nn:(ii+1)*nn] = np.real(dset[ii*nn:(ii+1)*nn, fid1:fid2, fid1:fid2])
                    
                Fid.create_dataset('d1', data=np.real(temp[fid1:fid2]))
                del Fid['d0']   
                
                
def chunked_prepare_delta(filename: str, props: dict, fullsum=1.0):
    """This function prepares an approximation of a delta function density profile and stores it inside a file. """
    nside = props['nside']
    mid = props['mid']
    rho0 = props['rho0']
    
    with h5py.File(filename, 'w') as F:
        dset = F.create_dataset('d1', (nside, nside, nside),
                                dtype='complex', chunks=True, maxshape=(nside, nside, nside))

        dset[mid, mid, mid] = fullsum*rho0/8.0
        dset[mid-1, mid, mid] = fullsum*rho0/8.0
        dset[mid, mid-1, mid] = fullsum*rho0/8.0
        dset[mid, mid, mid-1] = fullsum*rho0/8.0
        dset[mid-1, mid-1, mid] = fullsum*rho0/8.0
        dset[mid, mid-1, mid-1] = fullsum*rho0/8.0
        dset[mid-1, mid, mid-1] = fullsum*rho0/8.0
        dset[mid-1, mid-1, mid-1] = fullsum*rho0/8.0


def chunked_prepare_iso(filename: str, props: dict, slices=-1, debug=False):
    """ This function prepares an example of an isothermal density profile and stores it inside a file """
    nside = props['nside']
    rho0 = props['rho0']
    xs = props['xs']
    
    if slices == -1:
        slices = props['slices']
    
    nn = nside//slices
    show_progress = not debug
    
    if debug:
        print("Creating an isothermal profile...\n", flush=True)
    
    with h5py.File(filename, 'w') as F:
        dset = F.create_dataset('d1', (nside, nside, nside),
                                dtype='complex', chunks=True, maxshape=(nside, nside, nside))
        
        for ii in tqdm(range(slices), disable=show_progress):
            x_sub = xs[ii*nn:(ii+1)*nn]
            xx, yy, zz = np.meshgrid(x_sub, xs, xs, indexing='ij')
            
            dset[ii*nn:(ii+1)*nn] = 1.0/(1.0+xx**2+yy**2+zz**2)/rho0+0j


def chunked_prepare_gal(filename: str, props: dict, slices=-1, debug=False,
                        with_dm=False, with_bar=True, is_complex=True, spherical=False):
    """ This function prepares a density profile of MW and stores it inside a file.
    One can choose to turn on/off the baryon and the DM components as well as enforce a spherical boundary
    condition rho(R > boundary) = 0. """
    nside = props['nside']
    xs = props['xs']
    
    # Bulge properties
    par_blg = {"alpha": 1.8, "q": 0.5, "r0": 0.075, "rcut": 2.10, "rho0":  9.93e10}

    # Thin and Thick star disks
    par_sd = {"Rd_thin": 2.5, "Rd_thick": 3.0, "Sigma0_thin": 8.92e8, "Sigma0_thick": 1.83e8,
              "zd_thin": 0.3, "zd_thick": 0.9}
    # Gas disks
    par_gd = {"Sigma0_HI": 5.30e7, "Sigma0_H2": 2.18e9, "Rd_HI": 7.0, "Rd_H2": 1.5,
              "Rm_HI": 4.0, "Rm_H2": 12.0, "zd_HI": 0.085, "zd_H2": 0.045}

    # DM Halo (NFW)
    par_dm = {"rho0": 8.54e6, "rh": 19.6}

    if slices == -1:
        slices = props['slices']
    
    nn = nside//slices
    show_progress = not debug
    
    with h5py.File(filename, 'w') as F:
        if is_complex:
            dset = F.create_dataset('d1', (nside, nside, nside),
                                    dtype='complex', chunks=True, maxshape=(nside, nside, nside))
        else:
            dset = F.create_dataset('d1', (nside, nside, nside),
                                    dtype='float', chunks=True, maxshape=(nside, nside, nside))

        for ii in tqdm(range(slices), desc=f'Creating density profile [{filename}]', disable=show_progress):
            x_sub = xs[ii*nn:(ii+1)*nn]
            xx, yy, zz = np.meshgrid(x_sub, xs, xs, indexing='ij')
                        
            R = np.sqrt(xx**2 + yy**2)
            r = np.sqrt(xx**2 + yy**2 + zz**2)
            r_prime = np.sqrt(xx**2 + yy**2 + (zz/par_blg["q"])**2)

            if is_complex:
                density = 0.0+0.0j
            else:    
                density = 0.0
            
            if with_bar:
                density = density + par_blg["rho0"] * np.exp(-(r_prime/par_blg["rcut"])**2) / \
                          np.power((1.0+r_prime/par_blg["r0"]), par_blg["alpha"])
                density = density + par_sd["Sigma0_thin"]/(2.0*par_sd["zd_thin"]) *\
                    np.exp(-np.abs(zz)/par_sd["zd_thin"] - R/par_sd["Rd_thin"])
                density = density + par_sd["Sigma0_thick"]/(2.0*par_sd["zd_thick"]) *\
                    np.exp(-np.abs(zz)/par_sd["zd_thick"] - R/par_sd["Rd_thick"])
                density = density + par_gd["Sigma0_HI"]/(4.0*par_gd["zd_HI"]) *\
                    np.exp(- par_gd["Rm_HI"]/R - R/par_gd["Rd_HI"])/np.cosh(zz/(2.0*par_gd["zd_HI"]))**2
                density = density + par_gd["Sigma0_H2"]/(4.0*par_gd["zd_H2"]) *\
                    np.exp(- par_gd["Rm_H2"]/R - R/par_gd["Rd_H2"])/np.cosh(zz/(2.0*par_gd["zd_H2"]))**2

            if with_dm:
                density = density + par_dm["rho0"] * par_dm["rh"] / (r*(1+r/par_dm["rh"])**2)
            
            if spherical:
                mask = (r > xs[-1])
                density[mask] = 0*density[mask]

            dset[ii*nn:(ii+1)*nn] = density
            

def chunked_get_potential(filename: str, props: dict, slices=-1, debug=False, workers=4):
    """Computes the potential from a density stored in a file. Replaces the data in the file.
    The constant is ignored, in the sense that k^2 = 0 is regularized by manually setting
    (rho/k^2)[0,0,0] = 0. This means that the tilde{phi}[0,0,0], which corresponds to the constant
    shift, is just zero. How to recover this """

    ks = props['ks']
    nside = props['nside']
    G_Newton = props['G']
    
    if slices == -1:
        slices = props['slices']
    
    nn = nside//slices

    show_progress = not debug
    
    with h5py.File(filename, 'r+') as F:
        dset = F['d1']
        with tqdm(total=(5*slices), desc=f'Computing Potential [{filename}]', disable=show_progress) as pbar:
            for ii in range(slices):
                a = dset[:, ii*nn:(ii+1)*nn, :]
                dset[:, ii*nn:(ii+1)*nn] = fft.fft(a, axis=0, workers=workers)
                pbar.update(1)

            for ii in range(slices):
                ksub = ks[ii*nn:(ii+1)*nn]
                a = dset[ii*nn:(ii+1)*nn, :, :]
                a = fft.fft2(a, axes=(1, 2), workers=workers)
                pbar.update(1)
                kx, ky, kz = np.meshgrid(ksub, ks, ks, indexing='ij')
                ksq = np.pi * (kx ** 2 + ky ** 2 + kz ** 2)
                del kx, ky, kz
                # The last factor of pi comes from 4*pi/(2*pi)^2
                if ksq[0, 0, 0] == 0:
                    ksq[0, 0, 0] = 1
                a = a/ksq
                if ii == 0:
                    a[0, 0, 0] = 0
                # this last step could have been achieved by setting ksq[0,0,0] = +inf
                pbar.update(1)

                dset[ii*nn:(ii+1)*nn] = fft.ifft2(a, axes=(1, 2), workers=workers)
                pbar.update(1)

            for ii in range(slices):
                a = dset[:, ii*nn:(ii+1)*nn, :]
                dset[:, ii*nn:(ii+1)*nn] = -G_Newton*fft.ifft(a, axis=0, workers=workers)
                pbar.update(1)
                

def complex_2_real(filename: str, slices: int):
    """turns a file from complex to real, chunks"""

    temp_filename1 = filename.split('.')[0]+'_temp1.'+filename.split('.')[-1]
    
    with tqdm(total=2*slices, desc=f'Converting [{filename}] to a real array') as pbar:
    
        with h5py.File(filename, 'r+') as file_in:
            with h5py.File(temp_filename1, 'w') as file_out:
                dset_in = file_in['d1']
                nside = len(dset_in)
                nn = nside//slices
                dset_out = file_out.create_dataset('d1', (0, 0, 0),
                                                   dtype='float', chunks=True, maxshape=(nside, nside, nside))

                for ii in range(slices):
                    dset_out.resize(((ii+1)*nn, nside, nside))
                    dset_out[ii*nn:(ii+1)*nn] = np.real(dset_in[(slices-ii-1)*nn:(slices-ii)*nn])
                    dset_in.resize(((slices-ii-1)*nn, nside, nside))
                    pbar.update(1)

                del dset_in

        with h5py.File(filename, 'w') as file_out:
            with h5py.File(temp_filename1, 'r+') as file_in:
                dset_in = file_in['d1']
                dset_out = file_out.create_dataset('d1', (0, 0, 0),
                                                   dtype='float', chunks=True, maxshape=(nside, nside, nside))

                for ii in range(slices):
                    dset_out.resize(((ii+1)*nn, nside, nside))
                    dset_out[ii*nn:(ii+1)*nn] = np.real(dset_in[(slices-ii-1)*nn:(slices-ii)*nn])
                    dset_in.resize(((slices-ii-1)*nn, nside, nside))
                    pbar.update(1)

                del dset_in

    path = pathlib.Path(temp_filename1)
    path.unlink()

    
def real_2_complex(filename: str, slices: int):
    """Converts a real file into a complex one, haha"""
    
    temp_filename1 = filename.split('.')[0]+'_temp1.'+filename.split('.')[-1]
    
    with tqdm(total=2*slices, desc=f'Converting [{filename}] to a complex array') as pbar:
        with h5py.File(filename, 'r+') as file_in:
            with h5py.File(temp_filename1, 'w') as file_out:
                dset_in = file_in['d1']
                nside = len(dset_in)
                nn = nside//slices
                dset_out = file_out.create_dataset('d1', (0, 0, 0),
                                                   dtype='complex', chunks=True, maxshape=(nside, nside, nside))

                for ii in range(slices):
                    dset_out.resize(((ii+1)*nn, nside, nside))
                    dset_out[ii*nn:(ii+1)*nn] = dset_in[(slices-ii-1)*nn:(slices-ii)*nn]+0.0j
                    dset_in.resize(((slices-ii-1)*nn, nside, nside))
                    pbar.update(1)

                del dset_in

        with h5py.File(filename, 'w') as file_out:
            with h5py.File(temp_filename1, 'r+') as file_in:
                dset_in = file_in['d1']
                dset_out = file_out.create_dataset('d1', (0, 0, 0),
                                                   dtype='complex', chunks=True, maxshape=(nside, nside, nside))

                for ii in range(slices):
                    dset_out.resize(((ii+1)*nn, nside, nside))
                    dset_out[ii*nn:(ii+1)*nn] = dset_in[(slices-ii-1)*nn:(slices-ii)*nn]
                    dset_in.resize(((slices-ii-1)*nn, nside, nside))
                    pbar.update(1)

                del dset_in

    path = pathlib.Path(temp_filename1)
    path.unlink()


def sum_files(filename1: str, filename2: str, slices: int, a1=1.0, a2=1.0):
    """sums two files with weights a1, a2"""

    with h5py.File(filename1, 'r+') as F1:
        with h5py.File(filename2, 'r+') as F2:
            dset1 = F1['d1']
            dset2 = F2['d1']
            
            if len(dset1) != len(dset2):
                raise Exception(f"Cannot add: files {filename1}, {filename2} don't have the same size"
                                f" {len(dset1)} != {len(dset2)}!")
            nside = len(dset1)
            nn = nside//slices
            
            for ii in tqdm(range(slices)):
                dset1[ii*nn:(ii+1)*nn] = a1*dset1[ii*nn:(ii+1)*nn] + a2*dset2[ii*nn:(ii+1)*nn]
