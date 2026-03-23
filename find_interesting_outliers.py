# paste and run in python3 -i or a short script
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# load the final table (unchanged file name you used in plotting)
df = pd.read_csv("relicness_merged_with_stellar_complete_updated.csv")

# make sure required columns exist
for c in ("subhalo_id","stellar_mass_current","t_start","t50","t75","t90",
          "t95","t998","tfin","tfin_span","elem_Mg_mass","elem_Fe_mass",
          "stellar_halfmass_radius_kpc"):
    # HalfMassRadiusStars may have different name in your table; adapt if needed.
    pass

# If your radius column name differs use the correct name (e.g. 'r50' or 'HalfMassRadiusStars')
# Try some common options and pick the one that exists:
if "stellar_halfmass_radius_kpc" in df.columns:
    r_col = "stellar_halfmass_radius_kpc"
elif "r50" in df.columns:
    r_col = "r50"
elif "r50_proj" in df.columns:
    r_col = "r50_proj"
else:
    # if you actually stored log radius already, adapt here
    raise SystemExit("Can't find radius column; edit r_col to the correct name.")

# numeric conversions (safe)
df["stellar_mass_current"] = pd.to_numeric(df["stellar_mass_current"], errors="coerce")
df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
df["elem_Mg_mass"] = pd.to_numeric(df["elem_Mg_mass"], errors="coerce")
df["elem_Fe_mass"] = pd.to_numeric(df["elem_Fe_mass"], errors="coerce")
df["subhalo_id"] = df["subhalo_id"].astype(int)

# compute the same plotting quantities you used:
# stellar_mass_current in your table is in units of 1e10 Msun (you used SCALE=1e10)
# convert back to Msun before log if needed. Adapt as needed to match your plotting code.
# Here I assume stellar_mass_current is in 10^10 Msun like in compute script:
M10 = df["stellar_mass_current"].to_numpy(dtype=float)  # in 10^10 Msun
# convert to Msun:
M_msun = M10 * 1e10

# compute log10(M) and log10(R[kpc]) exactly as your plotting code
logM = np.log10(M_msun)
# your r50 in plotting was in kpc already -> verify; here assume r_col is in kpc
logR = np.log10(df[r_col].to_numpy(dtype=float))

# compute [Mg/Fe] exactly as in your plotting code
A_Mg = 24.305
A_Fe = 55.845
log_MgFe_sun = +0.10
Mg = df["elem_Mg_mass"].to_numpy(float)
Fe = df["elem_Fe_mass"].to_numpy(float)
with np.errstate(divide="ignore", invalid="ignore"):
    MgFe_number = (Mg / Fe) * (A_Fe / A_Mg)
    log10_number = np.where(MgFe_number > 0, np.log10(MgFe_number), np.nan)
    mgfe = log10_number - log_MgFe_sun

# attach to dataframe for easy selection
df["_logM"] = logM
df["_logR"] = logR
df["_mgfe"] = mgfe

# drop rows with missing coords (optional)
df_plot = df[np.isfinite(df["_logM"]) & np.isfinite(df["_logR"])].copy()
print("Rows usable for plotting:", len(df_plot))

# ---- Method A: box selection near the yellow cluster (bottom, just above logR~0.0)
# tweak these bounds to tighten/loosen selection
box_logR_low, box_logR_high = -0.05, 0.25   # "just above 0.0" -> adjust
box_logM_low, box_logM_high = 9.5, 11.0     # low mass side where those yellow points sit
# require high mgfe (top fraction), we pick the 95th percentile in this box
sub = df_plot[(df_plot["_logR"] >= box_logR_low) & (df_plot["_logR"] <= box_logR_high) &
              (df_plot["_logM"] >= box_logM_low) & (df_plot["_logM"] <= box_logM_high)]
if len(sub) == 0:
    print("No points found in that box; widen box bounds.")
else:
    mgfe_thresh = np.nanpercentile(sub["_mgfe"], 95)   # top 5% inside that box
    candidates = sub[sub["_mgfe"] >= mgfe_thresh].sort_values("_mgfe", ascending=False)
    print("Candidates in yellow-bottom region (by decreasing [Mg/Fe]):")
    print(candidates[["subhalo_id","_logM","_logR","_mgfe"]].head(20))
    # if you want the full list of IDs:
    ids_yellow = candidates["subhalo_id"].tolist()

# ---- Method B: strict selection for the green outlier between logM ≈ 11.0-11.5 AND logR < 0.5
mass_low, mass_high = 11.0, 11.5
logR_max = 0.5

# apply both conditions
sub2 = df_plot[(df_plot["_logM"] >= mass_low) & (df_plot["_logM"] <= mass_high) & (df_plot["_logR"] < logR_max)].copy()
print("Number in mass window and logR < 0.5:", len(sub2))

if len(sub2) == 0:
    print("No galaxies found with logM in [{:.2f},{:.2f}] and logR < {:.2f}.".format(mass_low, mass_high, logR_max))
    ids_green = []
else:
    # drop rows with NaN mgfe before computing stats
    sub2_valid = sub2[np.isfinite(sub2["_mgfe"])].copy()
    if len(sub2_valid) == 0:
        print("All candidates in the selection have NaN [Mg/Fe].")
        ids_green = sub2["subhalo_id"].tolist()   # return them anyway if you want
    else:
        # median in the selected bin
        med = float(np.nanmedian(sub2_valid["_mgfe"]))
        # robustly measure deviation from median and rank
        sub2_valid["mgfe_dev"] = np.abs(sub2_valid["_mgfe"] - med)
        outliers = sub2_valid.sort_values("mgfe_dev", ascending=False)

        # print top candidates (tweak n_print if needed)
        n_print = min(20, len(outliers))
        print("Top candidates in 11.0-11.5 mass bin with logR < 0.5 (by mgfe deviation):")
        print(outliers[["subhalo_id","_logM","_logR","_mgfe","mgfe_dev"]].head(n_print).to_string(index=False))

        # list of IDs to use for imaging (top N)
        N_select = 10
        ids_green = outliers["subhalo_id"].head(N_select).tolist()

# ids_green now contains the chosen subhalo_id candidates (possibly empty)
print("Selected ids_green (top):", ids_green)

# ---- Alternative precise: find points closest to visual coords on the plot
# e.g. if you saw a point near (logM_target, logR_target) use nearest-neighbour:
def find_nearest_ids(logM_target, logR_target, n=10):
    pts = np.column_stack((df_plot["_logM"].to_numpy(), df_plot["_logR"].to_numpy()))
    tree = cKDTree(pts)
    d, ix = tree.query([logM_target, logR_target], k=n)
    if np.isscalar(ix):
        ix = [ix]
        d = [d]
    rows = df_plot.iloc[ix]
    return rows[["subhalo_id","_logM","_logR","_mgfe"]], d

# example: near (10.5, 0.05) -> adjust to the visual position you want
rows_near, dists = find_nearest_ids(10.5, 0.05, n=10)
print("Nearest to (10.5, 0.05):")
print(rows_near)

# Plot their position in mass size plane

# Inputs
csv_in = "relicness_merged_with_stellar_complete_updated.csv"
outdir = "plots"
os.makedirs(outdir, exist_ok=True)

# load the final table
df = pd.read_csv(csv_in)

# find radius column
if "stellar_halfmass_radius_kpc" in df.columns:
    r_col = "stellar_halfmass_radius_kpc"
elif "r50" in df.columns:
    r_col = "r50"
elif "r50_proj" in df.columns:
    r_col = "r50_proj"
elif "HalfMassRadiusStars" in df.columns:
    r_col = "HalfMassRadiusStars"
else:
    raise SystemExit("Cannot find a radius column — edit r_col manually in the snippet.")

# numeric conversions (safe)
df["stellar_mass_current"] = pd.to_numeric(df["stellar_mass_current"], errors="coerce")
df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
df["elem_Mg_mass"] = pd.to_numeric(df["elem_Mg_mass"], errors="coerce")
df["elem_Fe_mass"] = pd.to_numeric(df["elem_Fe_mass"], errors="coerce")
df["subhalo_id"] = df["subhalo_id"].astype(int)

# compute plotting quantities (assumes stellar_mass_current in units of 1e10 Msun)
M10 = df["stellar_mass_current"].to_numpy(dtype=float)  # in 10^10 Msun
M_msun = M10 * 1e10
logM = np.log10(M_msun, where=(M_msun>0))
logR = np.log10(df[r_col].to_numpy(dtype=float), where=(df[r_col].to_numpy(dtype=float)>0))

# compute [Mg/Fe] (if needed)
A_Mg = 24.305
A_Fe = 55.845
log_MgFe_sun = +0.10
Mg = df["elem_Mg_mass"].to_numpy(float)
Fe = df["elem_Fe_mass"].to_numpy(float)
with np.errstate(divide="ignore", invalid="ignore"):
    MgFe_number = (Mg / Fe) * (A_Fe / A_Mg)
    log10_number = np.where(MgFe_number > 0, np.log10(MgFe_number), np.nan)
    mgfe = log10_number - log_MgFe_sun

# attach to dataframe
df["_logM"] = logM
df["_logR"] = logR
df["_mgfe"] = mgfe

# keep only rows with finite plotting coords
df_plot = df[np.isfinite(df["_logM"]) & np.isfinite(df["_logR"])].copy()
print("Rows usable for plotting:", len(df_plot))

# IDs you want to highlight
highlight_ids = [48958, 460550, 2022282, 18408, 682784, 2679391, 56334]

# --- base plot (match your original layout) ---
ztarget = 0.0   # minimal, self-contained; change if you want a different label
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 14
})
plt.figure(figsize=(8,6))

# background: all galaxies as faint grey (use df_plot values)
plt.scatter(df_plot["_logM"], df_plot["_logR"], alpha=0.7, s=10, color="lightgrey", label=f"Simulated galaxies at z={ztarget}")

# threshold line (Barro)
stellar_masses = np.logspace(9, 12, 100)
logsigma_ref = 10.0
plt.plot(
    np.log10(stellar_masses),
    (2/3)*(np.log10(stellar_masses) - logsigma_ref),
    linestyle='--',
    color='black',
    label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)'
)

# build lookup from df_plot for highlights
id_to_xy = dict(zip(df_plot["subhalo_id"].values, zip(df_plot["_logM"].values, df_plot["_logR"].values)))

# add markers (small, non-intrusive) and annotate
for sid in highlight_ids:
    if sid in id_to_xy:
        x, y = id_to_xy[sid]
        plt.scatter(
            x, y,
            s=50,               # modest marker size (you can reduce to 30 if too big)
            marker='o',
            facecolor='none',   # hollow marker
            edgecolor='black',
            linewidth=1.0,
            zorder=5
        )
        plt.annotate(str(sid), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=9)
    else:
        print("Highlight id not found in table or missing coords:", sid)

# layout identical to your base plot
plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
plt.ylabel(r"lg(Half Mass Radius / kpc)")
plt.title("Mass-size relation (COLIBRE 200m6)")
plt.legend(fontsize=8)
plt.grid(True)
plt.tick_params(axis='both', labelsize=12, direction='in', length=6, width=1)

# set x-limits exactly as in your earlier plot (9..12)
plt.xlim(9.0, 12.0)
plt.ylim(-0.5, 1.5)
# save to file (you asked not to show interactively; change if needed)
outpath = os.path.join(outdir, "marked_candidates_mass_size.png")
plt.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close()
print("Saved marked mass–size plot to:", outpath)