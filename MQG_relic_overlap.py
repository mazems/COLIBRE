import numpy as np
import pandas as pd

mqg = pd.read_csv("mqg_trackid_z2.0000_mcut10_ssfrcuttdep_sfrinst_ap50kpc_flagall_L0200N3008_Thermal.txt",
                  header=None, names=["track_id"])
z0 = pd.read_csv("z0_sags_trackids.csv")

mqg_ids = np.unique(mqg["track_id"].dropna().astype(np.int64))
z0_ids = np.unique(z0["track_id"].dropna().astype(np.int64))

common = np.intersect1d(mqg_ids, z0_ids)

overlap_df = pd.DataFrame({"track_id": common})
overlap_df.to_csv("mqg_sag_overlap_trackids.csv", index=False)

print("z=2 MQGs:", len(mqg_ids))
print("z=0 relics:", len(z0_ids))
print("Overlap:", len(common))
print("Fraction of z=2 MQGs that are z=0 relics:", len(common) / len(mqg_ids))