variables = {}

variables["ge3j_ge2t"] = [
    "MEM",
    "HT_tags",
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_untagged",
    "btagDiscriminatorAverage_tagged",
    "mass_tag_tag_min_deltaR",
    "mass_jet_tag_min_deltaR",
    "mass_tag_tag_max_mass",
    "mass_jet_jet_jet_max_pT",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_tag_tag",
    "minDeltaR_tag_tag",
    "maxDeltaEta_jet_jet",
    "maxDeltaEta_tag_tag",
    "H0_jet",
    "twist_tag_tag_max_mass",
    "multiplicity_higgsLikeDijet15"
    ]

variables["jets"]= [
    "pT_jet_jet_min_deltaR",
    "avgDeltaR_jet_jet",
    "avgDeltaR_jet_tag",
    "C_jet",
    "C_tag",
    "R1_tag",
    "mass_higgsLikeDijet",
    "minDeltaR_jet_jet",
    "R1_jet",
    "twist_jet_jet_max_mass",
    "mass_jet_jet_min_deltaR",
    "centrality_tags",
    "mass_higgsLikeDijet2",
    "circularity_jet",
    "centrality_jets_leps",
    "R4_tag",
    "R2_jet",
    "twist_jet_tag_max_mass",
    "aplanarity_jet",
    "twist_tag_tag_min_deltaR",
    "R4_jet",
    "H0_tag",
    "H3_tag",
    "jet1_pt",
    "jet1_eta",
    "jet1_phi",
    "jet1_M",
    "jet2_pt",
    "jet2_eta",
    "jet2_phi",
    "jet2_M",
    "jet3_pt",
    "jet3_eta",
    "jet3_phi",
    "jet3_M",
    "jet4_pt",
    "jet4_eta",
    "jet4_phi",
    "jet4_M",
    "jet1_btag",
    "jet2_btag",
    "jet3_btag",
    "jet4_btag",
    "bjet1_pt",
    "bjet1_eta",
    "bjet2_pt",
    "bjet2_eta",
    "bjet3_pt",
    "bjet3_eta",
    "bjet1_btag",
    "bjet2_btag",
    "bjet3_btag"
]

variables["ge4j_ge3t"] = [
    "MEM",
    "HT_tags",
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_untagged",
    "btagDiscriminatorAverage_tagged",
    "mass_tag_tag_min_deltaR",
    "mass_jet_tag_min_deltaR",
    "mass_tag_tag_max_mass",
    "mass_jet_jet_jet_max_pT",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_tag_tag",
    "minDeltaR_tag_tag",
    "maxDeltaEta_jet_jet",
    "maxDeltaEta_tag_tag",
    "H0_jet",
    "twist_tag_tag_max_mass",
    "multiplicity_higgsLikeDijet15"
    ]

variables["ge4j_ge4t"] = [
    "MEM",
    "HT_tags",
    "btagDiscriminatorAverage_tagged",
    "mass_tag_tag_min_deltaR",
    "mass_jet_tag_min_deltaR",
    "mass_jet_jet_jet_max_pT",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_tag_tag",
    "minDeltaR_tag_tag",
    "maxDeltaEta_jet_jet",
    "maxDeltaEta_tag_tag",
    "multiplicity_higgsLikeDijet15"
    ]

variables["ge4j_3t"] = [
    "MEM",
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_untagged",
    "btagDiscriminatorAverage_tagged",
    "mass_tag_tag_min_deltaR",
    "mass_tag_tag_max_mass",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_tag_tag",
    "minDeltaR_tag_tag",
    "maxDeltaEta_tag_tag",
    "H0_jet",
    "twist_tag_tag_max_mass"
    ]

variables["ge4j_2t"] = [
    "btagDiscriminatorAverage_untagged",
    "mass_tag_tag_min_deltaR",
    "mass_jet_tag_min_deltaR",
    "pT_tag_tag_min_deltaR",
    "pT_jet_jet_min_deltaR",
    "avgDeltaR_jet_jet",
    "avgDeltaR_jet_tag",
    "maxDeltaEta_jet_jet",
    "C_jet",
    "C_tag",
    "R1_tag",
    "multiplicity_higgsLikeDijet15"
    ]

variables["3j_3t"] = [
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_tagged",
    "mass_higgsLikeDijet",
    "mass_jet_jet_min_deltaR",
    "mass_tag_tag_max_mass",
    "pT_jet_jet_min_deltaR",
    "avgDeltaR_jet_jet",
    "minDeltaR_jet_jet",
    "maxDeltaEta_jet_jet",
    "C_jet",
    "R1_jet",
    "twist_jet_jet_max_mass"
    ]

variables["3j_2t"] = [
    "HT_tags",
    "ptSum_jets_leptons",
    "btagDiscriminatorAverage_untagged",
    "mass_jet_jet_min_deltaR",
    "mass_tag_tag_min_deltaR",
    "mass_jet_jet_jet_max_pT",
    "pT_tag_tag_min_deltaR",
    "avgDeltaR_jet_jet",
    "minDeltaR_jet_jet",
    "maxDeltaEta_jet_jet",
    "C_jet",
    "C_tag",
    "H0_jet"
    ]


all_variables = set( [v for key in variables for v in variables[key] ] )