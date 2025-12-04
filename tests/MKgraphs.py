from __future__ import annotations

import json
import os
import cloudpickle as pickle
import subprocess
import gzip
import zee_utils as zu
from coffea.dataset_tools import max_files, max_chunks, apply_to_fileset, filter_files, preprocess

import awkward as ak
import dask
import numpy as np
from dask.diagnostics import ProgressBar

import egamma_tnp
from egamma_tnp import ElectronTagNProbeFromNanoAOD


def assert_histograms_equal(h1, h2, flow):
    np.testing.assert_equal(h1.values(flow=flow), h2.values(flow=flow))
    assert h1.sum(flow=flow).value == h2.sum(flow=flow).value
    assert h1.sum(flow=flow).variance == h2.sum(flow=flow).variance


def assert_arrays_equal(a1, a2):
    assert sorted(a1.fields) == sorted(a2.fields)
    for i in a1.fields:
        assert ak.all(a1[i] == a2[i])


def test_cli():
    # fileset_path = "tests/NoAEGamma2023_fileset_grouped.json"
    # with open(fileset_path, "r") as f:
    #     fileset = json.load(f)
    
    # fileset_path = "tests/AllMC2023_Pre-BPix_fileset_ZeeHbb_xcache.json.gz"
    # with gzip.open(fileset_path, "r") as f:
    #     fileset = json.load(f)

    fileset_path = "tests/AllMC2023_Pre-BPix_fileset_ZeeHbb_xcache-Copy1.json-2.gz"
    with gzip.open(fileset_path, "r") as f:
        fileset = json.load(f)
        
    fileset = zu.change_sources(max_files(fileset,5))

    egamma_tnp.binning.set("el_eta_bins", [-2.5, -2.0, -1.566, -1.4442, -1.0, 0.0, 1.0, 1.4442, 1.566, 2.0, 2.5])

    workflow = ElectronTagNProbeFromNanoAOD(
        fileset=fileset,
        filters={"HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ": "HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"},
        filterbit={"HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ": 1},
        trigger_pt={"HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ": 12},
        tags_pt_cut=30,
        probes_pt_cut=7,
        tags_abseta_cut=2.5,
        probes_abseta_cut=2.5,
        cutbased_id="cutBased>=4",
        extra_zcands_mask=None,
        extra_filter=None,
        extra_filter_args={},
        use_sc_eta=False,
        use_sc_phi=False,
        avoid_ecal_transition_tags=False,
        avoid_ecal_transition_probes=False,
        require_event_to_pass_hlt_filter=False,
    )

    get_1d_pt_eta_phi_tnp_histograms_1_hlt = workflow.get_1d_pt_eta_phi_tnp_histograms(
        filter="HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        cut_and_count=False,
        mass_range=None,
        plateau_cut=None,
        eta_regions_pt=None,
        phi_regions_eta=None,
        eta_regions_phi=None,
        vars=["el_pt", "el_eta", "el_phi"],
        weight="weight",
        uproot_options={"allow_read_errors_with_report": True, "skipbadfiles": True, "timeout": 240},
    )

    to_compute = {
        "get_1d_pt_eta_phi_tnp_histograms_1_hlt": get_1d_pt_eta_phi_tnp_histograms_1_hlt
    }

    with ProgressBar():
        (out,) = dask.compute(to_compute)

    with open("tests/output/sample/get_nd_tnp_histograms_1/HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_histos.pkl", "wb") as f:
        pickle.dump(out, f)
    #os.system("rm -r tests/output")
