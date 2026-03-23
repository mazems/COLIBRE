# diagnostic_inspect.py
import numpy as np, h5py, os
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP

import utilities_statistics as us
from astropy.cosmology import Planck15
import astropy.units as u

MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'
VIRTUAL = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)
SOAP_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')

halo_id = 6041                # set the halo id you tested
MAPPING_NPZ = '/home/mzemsch/COLIBRE-analysis/ucmg_particle_index_mapping.npz'  # adjust

# load soap index map
sd = SWIFTDataset(SOAP_FILE)
soap_arr = np.array(sd.input_halos.halo_catalogue_index, dtype=np.int64)
soap_map = {int(val): int(idx) for idx,val in enumerate(soap_arr)}
soap_idx = soap_map[int(halo_id)]
print("soap_index for halo", halo_id, "=", soap_idx)

# build SWIFTGalaxy and read sg fields
sg = SWIFTGalaxy(VIRTUAL, SOAP(SOAP_FILE, soap_index=soap_idx))
print("sg.stars attributes available:", [a for a in dir(sg.stars) if not a.startswith('_')])

# masses via SG
m_sg_raw = sg.stars.masses
print("type(sg.stars.masses) =", type(m_sg_raw))
# if unyt-like, may have .units or .to
print("has 'units' attr?", hasattr(m_sg_raw, 'units'), "has 'to'?", hasattr(m_sg_raw,'to'))
m_sg = np.array(m_sg_raw, dtype=float)
print("sg: Nstars=", m_sg.size, "sum=", np.sum(m_sg))

# birth_scale_factors via SG -> compute lookback via us
if hasattr(sg.stars, 'birth_scale_factors'):
    a = np.array(sg.stars.birth_scale_factors, dtype=float)
    z = 1.0/a - 1.0
    tform_sg = np.array(us.look_back_time(z, h=0.6751, omegam=0.3121, omegal=0.6879), dtype=float)
    print("sg: tform (lookback) sample:", tform_sg[:5])
else:
    tform_sg = np.array([])

# Now: get particle indices to read HDF5 PartType4 arrays.
indices = None
if os.path.exists(MAPPING_NPZ):
    mp = np.load(MAPPING_NPZ, allow_pickle=False)
    key = str(int(halo_id))
    if key in mp.files:
        indices = mp[key].astype(np.int64)
        print("Got indices from mapping NPZ: N =", indices.size)
# fallback: try sg.stars.particle_ids
if indices is None:
    try:
        pids = np.array(sg.stars.particle_ids, dtype=np.int64)
        print("Got sg.stars.particle_ids N=", pids.size)
        # need to map particle ids -> indices using HDF5 ParticleIDs array
        with h5py.File(VIRTUAL, 'r') as fh:
            p4 = fh['PartType4']
            all_ids = np.array(p4['ParticleIDs'], dtype=np.int64)
        id_to_idx = {int(pid):int(i) for i,pid in enumerate(all_ids)}
        idxs = [id_to_idx.get(int(pid), -1) for pid in pids]
        if any(i<0 for i in idxs):
            print("Warning: some ids not found in snapshot ParticleIDs.")
        indices = np.array([i for i in idxs if i>=0], dtype=np.int64)
        print("Mapped particle_ids -> indices N=", indices.size)
    except Exception as e:
        print("No mapping available and no sg.stars.particle_ids mapping failed:", e)

# If we have indices, inspect HDF5 mass array
if indices is not None and indices.size>0:
    with h5py.File(VIRTUAL, 'r') as fh:
        p4 = fh['PartType4']
        # try several likely mass dataset names
        for name in ('InitialMasses','Masses','Masses0','masses'):
            if name in p4:
                print("Found mass dataset in HDF5:", name)
                m_h5_raw = p4[name]
                break
        else:
            m_h5_raw = None
        if m_h5_raw is not None:
            # read subset
            m_h5_sel = np.array(m_h5_raw[indices], dtype=float)
            print("h5: read N=", m_h5_sel.size, " sum=", np.sum(m_h5_sel))
            # check ratio to sg
            if m_sg.size==m_h5_sel.size:
                print("per-particle ratio sample:", (m_h5_sel[:10]/m_sg[:10]).tolist())
            print("sum(h5)/sum(sg) = ", np.sum(m_h5_sel)/np.sum(m_sg) if np.sum(m_sg)>0 else np.nan)
        else:
            print("No mass dataset found in HDF5 PartType4.")

    # compute tform from HDF5 birth scale factors (if present) in the h5 file and compare
    with h5py.File(VIRTUAL, 'r') as fh:
        p4 = fh['PartType4']
        if 'BirthScaleFactors' in p4:
            a_h5 = np.array(p4['BirthScaleFactors'][indices], dtype=float)
            z_h5 = 1.0/a_h5 - 1.0
            tform_h5_lb = np.array(us.look_back_time(z_h5, h=0.6751, omegam=0.3121, omegal=0.6879), dtype=float)
            # also compute cosmic age via Planck:
            tform_h5_age = np.array(Planck15.age(z_h5).to(u.Gyr).value, dtype=float)
            print("h5: birth a sample:", a_h5[:5])
            print("h5: tform lookback sample:", tform_h5_lb[:5])
            print("h5: tform age sample:", tform_h5_age[:5])
            # compare to sg tform (if same length)
            if tform_sg.size == tform_h5_lb.size:
                diffs = (tform_sg - tform_h5_lb)[:10]
                print("tform_sg - tform_h5_lb (first 10):", diffs)
        else:
            print("No BirthScaleFactors in HDF5 PartType4.")

else:
    print("Could not obtain particle indices to compare HDF5 vs SG.")