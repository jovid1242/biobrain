# Data validation — flywire_fafb_v783

**Overall: PASS** — {'PASS': 38, 'FAIL': 0, 'WARN': 3, 'INFO': 13}

Generated 2026-09-17T07:38:02+00:00 · commit `d47ea8ed10e91ecb256280e5181d697fe785df40` · Apple M5 · 5.5 s · peak RSS 3.5 GiB

Statuses: PASS = matches; FAIL = invariant broken or unexplained mismatch; WARN = known/explained difference or quality flag; INFO = observation.

| counts | value |
|---|---:|
| neurons | 139,255 |
| edges | 15,091,983 |
| rows | 16,847,997 |
| synapses | 54,492,922 |

## provenance

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| A1 | PASS | processed store was built from the raw files recorded in the lock file | [] | [] | lists raw file ids whose sha256 differs between lock and manifest |

## neurons

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| B1 | PASS | proofread neurons in release 783 | 139,255 | 139,255 | Dorkenwald et al. 2024, Nature 634:124-138, doi:10.1038/s41586-024-07558-y; "139,255 neurons" |
| B2 | PASS | root ids are unique | 139,255 | 139,255 |  |

## rows

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| C1 | PASS | rows in proofread_connections_783 (one per pre, post, neuropil) | 16,847,997 | 16,847,997 | Zenodo record 10676866 (file description; row count read from the Arrow footer by the research pass) |
| C2 | PASS | rows whose pre neuron is not a proofread neuron | 0 | 0 | Zenodo record 10676866 (file description; row count read from the Arrow footer by the research pass); the table is documented as the proofread subset |
| C3 | PASS | rows whose post neuron is not a proofread neuron | 0 | 0 | Zenodo record 10676866 (file description; row count read from the Arrow footer by the research pass) |
| C4 | PASS | rows with syn_count <= 0 | 0 | 0 | Zenodo record 10676866 (file description; row count read from the Arrow footer by the research pass); "one entry per neuron-neuron pair and neuropil if there is 1 or more synapses" |
| C5 | PASS | duplicate (pre, post, neuropil) rows | 0 | 0 |  |
| C6 | INFO | rows without a neuropil assignment (release label per spelling) | {"rows": 3193, "rows_by_label": {"UNASGD": 3193, "None": 0}, "synapses_by_label": {"UNASGD": 4185, "None": 0}, "null_rows": 0} |  | Dorkenwald 2024: synapses farther than 10 um from any neuropil 'were left unassigned'; the release spells this UNASGD here and 'None' in the per-neuron count files |
| C7 | PASS | distinct neuropil names in the connection table (unassigned label excluded) | 78 | 78 | Dorkenwald et al. 2024, Nature 634:124-138, doi:10.1038/s41586-024-07558-y; "78 fly brain regions known as neuropils" |
| C8 | PASS | predicted transmitter probabilities outside [0, 1] | 0 | 0 |  |
| C8b | INFO | rows without a transmitter prediction (all six probabilities NaN) | {"rows": 1732, "rows_partially_nan": 0, "synapses": 30806, "edges_without_any_prediction": 1251, "edges_partially_predicted": 481} |  | not described in the Zenodo record; excluded from the synapse-weighted mean of their edge |
| C9 | PASS | predicted rows whose six mean probabilities do not sum to 1 (|dev| > 1e-3) | 0 | 0 | max |sum - 1| = 2.59e-07 |

## graph

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| D1 | PASS | connected neuron pairs (no threshold) | 15,091,983 | 15,091,983 | Yu et al. 2025, bioRxiv doi:10.1101/2025.07.11.664377 (preprint; totals for the Buhmann synapse set); also Schlegel 2024: "139,255 nodes and around 15.1 million weighted edges" |
| D2 | PASS | synapses between proofread neurons | 54,492,922 | 54,492,922 | Yu et al. 2025, bioRxiv doi:10.1101/2025.07.11.664377 (preprint; totals for the Buhmann synapse set); also Dorkenwald 2024: "54.5 million synapses between these neurons" |
| D3 | PASS | neuron pairs with >= 5 synapses | 2,700,513 | 2,700,513 | Dorkenwald et al. 2024, Nature 634:124-138, doi:10.1038/s41586-024-07558-y; "2,700,513 such connections"; Lin et al. 2024 states 2,701,601 for v783 |
| D4 | PASS | neurons taking part in >= 5-synapse pairs | 134,181 | 134,181 | Dorkenwald et al. 2024, Nature 634:124-138, doi:10.1038/s41586-024-07558-y; "between 134,181 identified neurons" |
| D5 | INFO | self-connection pairs (pre == post) kept in the store | 0 |  | 0 synapses; not described in the sources; graph statistics drop them |
| D6 | INFO | neurons with no connection at all (isolated) | 616 |  |  |
| D7 | INFO | neurons with no outgoing / no incoming connection | {"no_out": 1250, "no_in": 2165} |  |  |
| D8 | PASS | independent re-derivation from raw (Arrow group_by) equals the processed store | {"rows": 16847997, "pairs": 15091983, "synapses": 54492922, "pairs_ge_5": 2700513, "neurons_in_pairs_ge_5": 134181, "self_connection_pairs": 0} | {"rows": 16847997, "pairs": 15091983, "synapses": 54492922, "pairs_ge_5": 2700513, "neurons_in_pairs_ge_5": 134181, "self_connection_pairs": 0} | different code path: hash aggregation on root ids, no index mapping |

## store

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| E1 | PASS | out_indptr starts at 0, ends at E, non-decreasing | True | True |  |
| E2 | PASS | out_indices within [0, N) | True | True |  |
| E3 | PASS | post ids strictly increasing inside every CSR row (no duplicate pairs) | 0 | 0 |  |
| E4 | PASS | CSC is an exact permutation of CSR (post and pre agree for every slot) | True | True |  |
| E5 | PASS | rows sorted by edge, cover every edge, and their synapses sum to the edge count | True | True |  |
| E6 | PASS | quantized transmitter probabilities of a predicted edge sum to 255 +/- 3 (rounding) | 0 | 0 |  |
| E6b | PASS | edges stored without prediction (all zeros) = edges without any predicted row | 1,251 | 1,251 |  |
| E7 | PASS | every array matches its manifest sha256 | [] | [] |  |

## completeness

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| F1a | PASS | pre-side synapse counts summed over all segments (incl. fragments) | 130,054,535 | 130,054,535 | Zenodo record 10676866 (file description; row count read from the Arrow footer by the research pass); 130,054,535 = rows of flywire_synapses_783 (research pass); Dorkenwald: '~130 million synapses' |
| F1b | PASS | pre-side synapses assigned to one of the neuropils (all segments) | 130,038,118 | 130,038,118 | Codex neuropil_stats.csv for v783 (totals measured by the research pass; file not part of this pipeline) |
| F1c | PASS | synapses in a neuropil whose presynaptic side lies on a proofread neuron | 121,904,312 | 121,904,312 | Codex neuropil_stats.csv for v783 (totals measured by the research pass; file not part of this pipeline); 121,914,213 including unassigned; attachment rates 93.7 % pre / 44.7 % post (Dorkenwald 2024) |
| F1d | PASS | neurons whose pre-side synapses inside the proofread graph exceed their all-partner total | 0 | 0 |  |
| F1e | INFO | share of a neuron's pre-side synapses whose partner is a proofread neuron (percentiles) | {"p5": 0.2438, "p25": 0.4023, "p50": 0.5199, "p75": 0.6273, "p95": 0.7778} |  | the remainder goes to unproofread fragments; this is data the graph does not contain |
| F1f | PASS | duplicate (neuron, neuropil) rows in the pre count file | 0 | 0 |  |
| F2a | PASS | post-side synapse counts summed over all segments (incl. fragments) | 130,054,535 | 130,054,535 | Zenodo record 10676866 (file description; row count read from the Arrow footer by the research pass); 130,054,535 = rows of flywire_synapses_783 (research pass); Dorkenwald: '~130 million synapses' |
| F2b | PASS | post-side synapses assigned to one of the neuropils (all segments) | 130,038,118 | 130,038,118 | Codex neuropil_stats.csv for v783 (totals measured by the research pass; file not part of this pipeline) |
| F2c | PASS | synapses in a neuropil whose postsynaptic side lies on a proofread neuron | 58,096,025 | 58,096,025 | Codex neuropil_stats.csv for v783 (totals measured by the research pass; file not part of this pipeline); 58,101,724 including unassigned; attachment rates 93.7 % pre / 44.7 % post (Dorkenwald 2024) |
| F2d | PASS | neurons whose post-side synapses inside the proofread graph exceed their all-partner total | 0 | 0 |  |
| F2e | INFO | share of a neuron's post-side synapses whose partner is a proofread neuron (percentiles) | {"p5": 0.719, "p25": 0.9, "p50": 0.9368, "p75": 0.9597, "p95": 0.9869} |  | the remainder goes to unproofread fragments; this is data the graph does not contain |
| F2f | PASS | duplicate (neuron, neuropil) rows in the post count file | 0 | 0 |  |

## annotations

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| G1 | WARN | annotation rows (v3.1.0) vs proofread neurons | 139,248 | 139,255 | 14 neurons have no row; see G1b |
| G2 | WARN | annotation rows whose root id is not in the 783 proofread-neuron list | 7 | 0 | the repository README states root ids are from release 783; such rows cannot be joined to the graph and are dropped. Ids: [720575940628215433, 720575940624154153, 720575940622062913, 720575940618455453, 720575940616847814, 720575940637590670, 720575940626850833] |
| G3 | PASS | neurons with more than one annotation row | 0 | 0 |  |
| G4 | INFO | fraction of neurons with a non-empty value, per column | {"supervoxel_id": 0.9999, "pos_x": 0.9999, "pos_y": 0.9999, "pos_z": 0.9999, "soma_x": 0.8481, "soma_y": 0.8481, "soma_z": 0.8481, "nucleus_id": 0.7689, "flow": 0.9999, "super_class": 0.9999, "cell_class": 0.772, "cell_sub_class": 0.1855, "supertype": 0.243, "cell_type": 0.9889, "hemibrain_type": 0.2389, "ito_lee_hemilineage": 0.2696, "hartenstein_hemilineage": 0.2499, "top_nt": 0.9956, "top_nt_conf": 0.9956, "known_nt": 0.6307, "known_nt_source": 0.6307, "side": 0.9999, "nerve": 0.0693, "status": 0.0047, "dimorphism": 0.9999, "fru_dsx": 0.0238} |  |  |
| G5 | INFO | super_class counts (v3.1.0) | {"": 14, "ascending": 1750, "central": 32383, "descending": 1303, "endocrine": 80, "motor": 110, "optic": 77537, "sensory": 16904, "sensory_ascending": 612, "visual_centrifugal": 524, "visual_projection": 8038} |  |  |

## annotations-paper

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| H1 | PASS | v2.1.0 rows / rows that are proofread neurons / unique | [139255, 139255, 139255] | [139255, 139255, 139255] |  |
| H2 | PASS | v2.1.0 super_class counts vs the counts printed in Schlegel 2024 | {"central": 32388, "optic": 77536, "visual_projection": 8053, "visual_centrifugal": 524, "sensory (non-visual)": 5512, "ascending": 2362, "descending": 1303, "endocrine": 80, "motor": 106} | {"central": 32388, "optic": 77536, "visual_projection": 8053, "visual_centrifugal": 524, "sensory (non-visual)": 5512, "ascending": 2362, "descending": 1303, "endocrine": 80, "motor": 106} | Schlegel et al. 2024, Nature 634:139-152, doi:10.1038/s41586-024-07686-5; sensory neurons of cell_class visual/ocellar (11,391) are not part of the printed sentence |
| H3 | PASS | v2.1.0 distinct types, counting cell_type or else hemibrain_type | 8,453 | 8,453 | Schlegel et al. 2024, Nature 634:139-152, doi:10.1038/s41586-024-07686-5; "8,453 annotated cell types" = 3,643 hemibrain-derived + 4,581 new + 229 other; cell_type alone has 5,634 distinct values |
| H4 | WARN | v2.1.0 share of neurons with cell_type or hemibrain_type | 0.9465 | 0.964 | Schlegel et al. 2024, Nature 634:139-152, doi:10.1038/s41586-024-07686-5; "cell types for 96.4% of all neurons"; the denominator behind 96.4 % is not stated, not resolved here |

## annotations

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| G1b | INFO | neurons without a v3.1.0 row, as annotated in v2.1.0 | [{"root_id": 720575940633242449, "super_class": "central", "cell_class": "CX", "cell_type": "CB.FB3,4A2"}, {"root_id": 720575940628159081, "super_class": "sensory", "cell_class": "mechanosensory", "cell_type": "JO-B"}, {"root_id": 720575940626741265, "super_class": "sensory", "cell_class": "mechanosensory", "cell_type": "JO-B"}, {"root_id": 720575940636488667, "super_class": "sensory", "cell_class": "mechanosensory", "cell_type": "JO-B"}, {"root_id": 720575940625523749, "super_class": "optic", "cell_class": "LOP", "cell_type": ""}, {"root_id": 720575940639387763, "super_class": "optic", "cell_class": "LOP", "cell_type": ""}, {"root_id": 720575940642560987, "super_class": "optic", "cell_class": "LA>ME", "cell_type": "L5"}, {"root_id": 720575940607862706, "super_class": "optic", "cell_class": "LO>ME", "cell_type": "CB3834"}, {"root_id": 720575940613437206, "super_class": "optic", "cell_class": "LO", "cell_type": ""}, {"root_id": 720575940606792881, "super_class": "optic", "cell_class": "ME>LA", "cell_type": "T1"}, {"root_id": 720575940623913933, "super_class": "optic", "cell_class": "LO", "cell_type": ""}, {"root_id": 720575940623450077, "super_class": "optic", "cell_class": "LO", "cell_type": ""}, {"root_id": 720575940619169349, "super_class": "optic", "cell_class": "ME>LO", "cell_type": "Tm20"}, {"root_id": 720575940613336473, "super_class": "optic", "cell_class": "ME", "cell_type": ""}] |  |  |

## distributions

| id | status | check | observed | expected | source / note |
|---|---|---|---|---|---|
| I1 | INFO | synapses per connected pair (histogram) | {"1": 7496016, "2": 2679736, "3": 1379004, "4": 836714, "5-9": 1633691, "10-19": 699958, "20-49": 300426, "50-99": 50601, "100-999": 15810, ">=1000": 27} |  |  |
| I2 | INFO | synapses per pair: mean / median / p99 / max | {"mean": 3.611, "median": 2.0, "p99": 32.0, "max": 2405} |  |  |
| I3 | INFO | synapses between proofread neurons per neuropil (top 10; '' = unassigned) | {"ME_R": 8821540, "ME_L": 7712403, "LO_R": 3877311, "LO_L": 3792886, "GNG": 2713213, "AVLP_R": 2000857, "LOP_R": 1901109, "AVLP_L": 1510021, "LOP_L": 1189689, "SMP_R": 898333} |  |  |
