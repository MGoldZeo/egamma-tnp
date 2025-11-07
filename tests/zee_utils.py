import awkward as ak
from dataclasses import dataclass
from functools import reduce
import gzip
import json
import numpy as np
import operator
import cloudpickle as cp
import vector
import warnings
import dask
from coffea.analysis_tools import PackedSelection
try:
    from cowtools import combine_rename_results, scale_results
except ImportError:
    warnings.warn("""Either cowtools is not installed here, or it does not have one of the
                     requested functions 'combine_rename_results' or 'scale_results'. Certain
                     classes, at least 'XSecScaler', may not work.""")

#Values obtained from jet_assignment/zfit_ZH_reco_mass.ipynb
MU_Z = 96.3478
SIGMA_Z = 14.3686
Z_MASS = 91.1880
MU_H = 131.116
SIGMA_H = 18.801

Z_MASS = 91.1880
H_MASS = 125.35

#Values obtained from jet_assignment/zfit_ZH_mass_dif.ipynb
MU_DIFF = 34.910

#Physics values (2023 Pre-BPix)
BT_tight = 0.6172 #PNet b-tagging tight WP for 2023 Pre-BPix
BT_med = 0.1917
BT_loose = 0.0358

DEFAULT_GROUPING_MAP = {
    "QCD": lambda dset: dset.startswith("/QCD"),
    "ZJets": lambda dset: dset.startswith("/DY"),
    "WJets": lambda dset: dset.startswith("/WJets"),
    "ttbar": lambda dset: dset.startswith("/TTto"),
    "SingleTop": lambda dset: dset.startswith("/TWminus") or dset.startswith("/TbarWplus"),
    "Diboson": lambda dset: dset.startswith("/WW") or dset.startswith("/ZZ") or dset.startswith("/WZ"),
    "ZeeHbb": lambda dset: dset.startswith("/ZH"),
    "ggZeeHbb": lambda dset: dset.startswith("/ggZH"),
}

year = "2024"   # change this once
runs = list("ABCDEFGHIJ")  # A–J
streams = ("EGamma0", "EGamma1")

for run in runs:
    for stream in streams:
        key = f"{stream}{run}"
        DEFAULT_GROUPING_MAP[key] = lambda dset, s=stream, r=run: dset.startswith(f"/{s}/Run{year}{r}")


def empty_ele(events):
    e = events.Electron[0]
    return ak.zeros_like(e)

def empty_jet(events):
    e = events.Jet[0]
    return ak.zeros_like(e)

def apptrigs(events):
    triggers = PackedSelection()
    #if hasattr(events.HLT, "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL"):
        #triggers.add("ETrig1", events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL)
    #if hasattr(events.HLT, "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"):
        #triggers.add("ETrig2", events.HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ)
    #if hasattr(events.HLT, "DoubleEle24_eta2p1_WPTight_Gsf"):
        #triggers.add("ETrig3", events.HLT.DoubleEle24_eta2p1_WPTight_Gsf)
    #if hasattr(events.HLT, "Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30"):
        #triggers.add("ETrig4", events.HLT.Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30)
    triggers.add("ETrig1", events.HLT.Ele32_WPTight_Gsf)
    return triggers.any(*triggers.names) if triggers.names else ak.zeros_like(events.event, dtype=bool)

def sel_jet(jets):
    try:
        jetid_pass = jets.jetId >= 6
    except AttributeError:
        jetid_pass = ak.ones_like(jets.pt, dtype=bool)  # pass all jets

    select = jetid_pass & (jets.pt >= 20) & (jets.pt <= 100) & (abs(jets.eta) <= 2.5)
    return select

def sel_jet_events(events):
    jet1 = events.Jet[:,0]
    jet2 = events.Jet[:,1]
    try:
        jetid_pass = ((jet1.jetId==6) | (jet1.jetId==2)) & ((jet2.jetId==6) | (jet2.jetId==2))
    except AttributeError:
        jetid_pass = ak.ones_like(jets.pt, dtype=bool)  # pass all jets

    select_j1 = ((jet1.pt >= 7) & (jet1.pt <= 100) & (abs(jet1.eta) <= 2.5)) 
    select_j2 = ((jet2.pt >= 7) & (jet2.pt <= 100) & (abs(jet2.eta) <= 2.5))
    select = jetid_pass & select_j1 & select_j2
    return select
    
def sel_ele(electrons, tight=False):
    if tight == True:
        select = (10 <= electrons.pt) & (electrons.pt <= 90) & (abs(electrons.eta) <= 2.4) & (electrons.cutBased >= 4) & (electrons.pfRelIso03_all < 0.05) #eta DY->2.4, Hbb->2.5, iso requirement
    else:
        select = (7 <= electrons.pt) & (electrons.pt <= 100) & (abs(electrons.eta) <= 2.5) & (electrons.cutBased >= 3) & (electrons.pfRelIso03_all < 0.1) #eta DY->2.4, Hbb->2.5, iso requirement
    return select

def sel_ele_events(events):
    e1 = events.Electron[:,0] 
    e2 = events.Electron[:,1]
    if not ((events.Electron is None) or len(events.Electron.fields) == 0):
        select_e1 = ((e1.cutBased >= 4) & (7 <= e1.pt) & (e1.pt <= 100) & (abs(e1.eta) <= 2.4) & (e1.pfRelIso03_all < 0.10))
        select_e2 = ((e2.cutBased >= 4) & (7 <= e2.pt) & (e2.pt <= 100) & (abs(e2.eta) <= 2.4) & (e2.pfRelIso03_all < 0.10))
        select = select_e1 & select_e2
        return select

def dijet_bTag(events, btag_wp=BT_tight):
    jets = events.Jet
    bjetscores = ak.sort(jets.btagPNetB, ascending=False)
    return ((bjetscores[:,0]>btag_wp) & (bjetscores[:,1]>btag_wp))
'''
def sel_ele(events):
    electrons = events.Electron
    if not ((electrons is None) or len(electrons.fields) == 0):
        select = (electrons.cutBased >= 4) & (7 <= electrons.pt) & (electrons.pt <= 100) & (abs(electrons.eta) <= 2.5)
    return ak.any(select, axis=1)
'''   
def regress_btagged(jets,btag_wp=BT_loose):
    #From Ryan's code
    if jets is None or len(jets.fields) == 0:
        return ak.ones_like(jets)
    elif not hasattr(jets, "btagPNetB"):
        return jets
    filt = jets.btagPNetB >= btag_wp
    #If we multiply jets directly, we lose b-tagging fields. So instead, regress pts and reattach at end
    reg_pt = ak.where(filt,jets.pt * jets.PNetRegPtRawCorr * jets.PNetRegPtRawCorrNeutrino,jets.pt)
    #Create output array
    out_jets = jets
    out_jets["pt"] = reg_pt
    return out_jets

def remove_overlap(electrons, jets):
    cross = ak.cartesian({"ele":electrons, "jets":jets}, axis=1)
    dr = cross.ele.delta_r(cross.jets)
    return dr
    
    
def sgl_baseline(jets, electrons, btag_wp=BT_tight):

    trash, electrons, jets = difilt(electrons=electrons,jets=jets)
    
    #Apply offline selections
    sglsel = PackedSelection()

    #Opposite sign selection
    oppsign = electrons[:,0].charge != electrons[:,1].charge
    sglsel.add("OppSign",oppsign)

    #Z mass filt
    dielectron = electrons[:,0] + electrons[:,1]
    zmassfilt = (dielectron.mass > 70) & (dielectron.mass < 110)
    sglsel.add("ZMFilt",zmassfilt)
    
    #Find b-tagged jets
    bjets = jets[jets.btagPNetB >= btag_wp]
    sglsel.add("Nbtag",ak.num(bjets) >= 2)

    #Clear overlap
    overlap = remove_overlap(electrons, jets)
    overlapfilt = ak.all(overlap > 0.4, axis=1)
    sglsel.add("Overlap",overlapfilt)

    return sglsel.all(*sglsel.names)

def sgl_t1(jets, electrons, btag_wp=BT_tight):
        
    trash, electrons, jets = difilt(electrons=electrons,jets=jets)
    
    #Apply offline selections
    sglsel = PackedSelection()
    
    #pt selection
    sglsel.add("E1pt",electrons[:,0].pt >= 22)
    sglsel.add("E2pt",electrons[:,1].pt >= 11)
    sglsel.add("J1pt",jets[:,0].pt >= 30)
    sglsel.add("J2pt",jets[:,1].pt >= 30)

    #Tighter Z mass filt
    dielectron = electrons[:,0] + electrons[:,1]
    zmassfilt = (dielectron.mass > 85) & (dielectron.mass < 95)
    sglsel.add("ZMFilt",zmassfilt)  

    #Tighten b-tag score
    trash, trash1, jets = difilt(electrons=electrons, jets=jets)
    bjetscores = ak.sort(jets.btagPNetB, ascending=False)
    avg_btag = (bjetscores[:,0] + bjetscores[:,1])/2
    sglsel.add("Tbtag",avg_btag > .7)

    return sglsel.all(*sglsel.names)


def scale_combine_mc_only(mc,fs_mc,lumi,verbose=False,dont_scale=None,grouping_map=DEFAULT_GROUPING_MAP):
    '''
    Inputs:
        mc: (dict) Values are dicts containing hists, floats, etc. to be scaled. If str, path to the pickled dict file.
        fs_mc: (dict | str) Dictionary with fs_mc[dset]["metadata"]["xsec"] the cross section values to scale processes
                to, and fs_mc[dset]["metadata"]["short_name"] (optional). If str, path to the gzipped json file
                containing the dict.
        lumi: (float | int) Luminosity to scale to, in the same units as fs_mc's xsecs.
        verbose: (bool, optional) If True, print the MC lumi-scaling weight for each dataset
        dont_scale: (list[str], optional) An iterable of things to not scale. These are keys in the dicts that are
                themselves values of mc. If not given, defaults to whatever the default of scale_results is.
    '''
    if type(mc) == str:
        with open(mc, 'rb') as f:
            mc = cp.load(f)
    if type(fs_mc) == str:
        with gzip.open(fs_mc) as f:
            fs_mc = json.load(f)
    
    #Scale MC to lumi = LUMI (top of notebook)
    mc_xsecs = {}
    mc_evt_cnts = {}
    for dset in mc.keys():
        mc_xsecs[dset] = fs_mc[dset]["metadata"]["xsec"]
        mc_evt_cnts[dset] = mc[dset]["RawEventCount"]
    if dont_scale is None:
        mc_scaled = scale_results(mc,lumi,mc_xsecs,mc_evt_cnts,verbose=verbose)
    else:
        mc_scaled = scale_results(mc,lumi,mc_xsecs,mc_evt_cnts,verbose=verbose,dont_scale=dont_scale)
    
    #Regroup and rename datasets in outputs
    short_name_map = {}
    for dset in mc_scaled.keys():
        short_name_map[dset] = fs_mc[dset]["metadata"].get("short_name",dset)
    return combine_rename_results(mc_scaled,grouping_map=grouping_map,short_name_map=short_name_map)


def scale_results(mc,lumi,mc_xsecs,mc_evt_cnts,verbose=False,dont_scale=["RawEventCount", "numlist"]):
    '''
    Inputs:
        mc: (dict) Values are dicts containing hists, floats, etc. to be scaled
        lumi: (float | int) Luminosity to scale to
        mc_xsecs: (dict) Map keys from mc to their cross sections
        mc_evt_cnts: (dict) Map keys from mc to their raw event counts
        dont_scale: (iterable, optional) An iterable of things to not scale. These are keys
            in the dicts that are themselves values of mc.

    Outputs:
        Hist with the same structure as mc, except results are scaled to lumi according to
        mc_xsecs and mc_evt_cnts
    '''
    #If mc is a string, treat as filepath and retrieve results
    if type(mc) == str:
        #If strings end in .pkl, use pickle to load
        if not mc.endswith(".pkl"):
            warnings.warn(f"""
                            Warning: arg mc is {mc}, and is type str, but does not end in '.pkl'. Trying to read with pickle
                            anyway. If this is not desired, please retrieve the hist of results and pass that as mc arg.""")
        with open(mc,"rb") as f:
            mc = cp.load(f)

    out = {}
    for dset,results in mc.items():
        out[dset] = {}
        mc_factor = mc_xsecs[dset]*lumi/mc_evt_cnts[dset]
        if verbose:
            print(f"Dataset {dset} has MC lumi-scaling weight {mc_factor}")
        for obs_name, obs in results.items():
            if obs_name not in dont_scale:
                out[dset][obs_name] = obs * mc_factor
            else:
                out[dset][obs_name] = obs

    return out

def clean_array(*arrays):
    """Drop None entries across all arrays."""
    flattened = []
    for arr in arrays:
        try:
            arr = ak.flatten(arr)
        except Exception:
            pass
        flattened.append(arr)

    # Compute mask: True where none of the arrays are None
    masks = [~ak.is_none(arr) for arr in flattened]
    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask = combined_mask & m

    # Apply mask and drop any remaining None
    result = []
    for arr in flattened:
        arr_masked = arr[combined_mask]
        result.append(ak.drop_none(arr_masked))

    return tuple(result)

def chi2_mass(jets, electrons, output_score=False, btag_min=-2.0, sub_btag_min=-2.0):
    '''
    Assign two jets to Higgs (bb), and use the provided electrons for Z (ee),
    minimizing chi-squared = (m_H - MU_H)^2 / SIGMA_H^2 + (m_Z - MU_Z)^2 / SIGMA_Z^2.

    Parameters
    ----------
    jets : ak.Array
        Array of jets with length ≥ 2 per event. Must include btagPNetB and 4-vector fields.
    electrons : ak.Array
        Array of exactly two electrons per event. Must include 4-vector fields.
    output_score : bool
        If True, returns chi-squared scores.
    btag_min : float
        At least one Higgs jet must pass this b-tag value.
    sub_btag_min : float
        The second Higgs jet must pass this minimum b-tag.

    Returns
    -------
    Tuple (H1, H2, Z1, Z2) if output_score is False,
    or (H1, H2, Z1, Z2, chi2) if output_score is True.
    '''
    if (hasattr(jets, "btagPNetB") & (not (jets is None)) & (not (electrons is None))):
        # Get all possible jet pairs
        combs = ak.combinations(jets, 2, fields=["H1", "H2"])

        H1 = combs["H1"]
        H2 = combs["H2"]

        # B-tag requirement
        btag_pass = ((H1.btagPNetB >= btag_min) & (H2.btagPNetB >= sub_btag_min)) | \
                ((H1.btagPNetB >= sub_btag_min) & (H2.btagPNetB >= btag_min))

        H1 = H1[btag_pass]
        H2 = H2[btag_pass]

        # Electron pairing (Z candidate)
        Z = electrons[:, 0] + electrons[:, 1]
        Z = ak.broadcast_arrays(Z, H1)[0]  # match dimensions with H candidates

        # Higgs pairing
        H = H1 + H2

        # Chi-squared computation
        chi2 = ((Z.mass - MU_Z) ** 2) / (SIGMA_Z ** 2) + ((H.mass - MU_H) ** 2) / (SIGMA_H ** 2)

        # Minimize
        best_idx = ak.singletons(ak.argmin(chi2, axis=-1))

        out_H1 = ak.firsts(H1[best_idx])
        out_H2 = ak.firsts(H2[best_idx])
        out_Z1 = ak.firsts(ak.broadcast_arrays(electrons[:,0], H1)[0][best_idx])
        out_Z2 = ak.firsts(ak.broadcast_arrays(electrons[:,1], H1)[0][best_idx])
        

        if output_score:
            best_chi2 = ak.firsts(chi2[best_idx])
            return out_H1, out_H2, out_Z1, out_Z2, best_chi2
        else:
            return out_H1, out_H2, out_Z1, out_Z2

def xcacheify(fname,xrd_base="root://cmsxcache.hep.wisc.edu/"):
    fparts = fname.split('/store')
    if len(fparts) != 2:
        raise ValueError(f"Filepath {fname} does not fit pattern (splittable on '/store'")
    xfname = '/store'.join([xrd_base,fparts[-1]])

    return xfname

def change_sources(fileset_in,new_source="root://cmsxcache.hep.wisc.edu/"):
    fileset_out = {}
    for dset in fileset_in:
        dataset_ready = {}
        for key, val in fileset_in[dset].items():
            if key != 'files':
                dataset_ready[key] = val
            else:
                files_info = {}
                for fkey, fval in val.items():
                    xfname = xcacheify(fkey,xrd_base=new_source)
                    files_info[xfname] = fval
                dataset_ready[key] = files_info
        fileset_out[dset] = dataset_ready

    return fileset_out

def difilt(events=None, electrons=None, jets=None):
    elenum = ak.num(electrons)
    jetnum = ak.num(jets)
    difilter = (elenum >= 2) & (jetnum >= 2)
    if events is not None:
        events = events[difilter]
    if electrons is not None:
        electrons = electrons[difilter]
    if jets is not None:
        jets = jets[difilter]
    return events, electrons, jets

@dask.delayed
def dp(arr, label=""):
    print(f"{label}: {ak.to_list(arr)}")
    return None

def make_bin_edges(num_bins, start, stop):
    return np.linspace(start, stop, num_bins + 1)

def bin_values(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    errors = np.sqrt(counts)
    return counts, errors

def dR_Mass(a, b):
    dphi = abs((a.phi - b.phi + np.pi) % (2*np.pi) - np.pi)
    deta = abs(a.eta - b.eta)
    dr = np.sqrt(deta**2 + dphi**2)
    dpt = abs(a.pt - b.pt)
    return dr, dpt

def align_arrays(a,b):
    a=a[ak.num(a)>=1]
    a=a[ak.num(b)>=1]
    b=b[ak.num(a)>=1]
    b=b[ak.num(b)>=1]
    return(a,b)
            
def inv_Mass(a,b):
    return (np.sqrt(2*a.pt*b.pt*(np.cosh(a.eta-b.eta)-np.cos(a.phi-b.phi))))

def z_mass_metric_soft(a, b):
    mass_diff = abs(inv_Mass(a,b) - Z_MASS)
    return mass_diff




