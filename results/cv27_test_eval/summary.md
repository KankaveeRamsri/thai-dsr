# CV 27.0 test evaluation (924 rows per system)

Mean [95% CI], CI = percentile bootstrap over speakers (client_id), 10000 resamples, seed 42.

## Systems and timelines

| system | role | timeline | inference | checkpoint |
|---|---|---|---|---|
| clean | reference: untouched clean audio (ASR only) | clean | the untouched clean clip (ASR reference only) | – |
| distorted | floor | distorted | the distorted input itself | – |
| pv_roundtrip | control: inverse warp only | distorted | clean stretched onto the distorted timeline (phase vocoder), no vocoder | – |
| copysynth_clean | ceiling (clean timeline) | clean | clean mel -> HiFi-GAN, on the clean timeline | – |
| copysynth_tempo | ceiling (distorted timeline) | distorted | clean mel warped onto the distorted timeline -> HiFi-GAN | – |
| mapper_linear_oracle | (a) linear, clean length (oracle) | clean | linear model; embedding interpolated to the CLEAN mel frame count, as in training (uses the clean duration as an oracle) | epoch 10, best_epoch 10, sha256 11912f0ec31f |
| mapper_linear_deploy | (a) linear, distorted length (deployable) | distorted | linear model; embedding interpolated to the DISTORTED mel frame count (no oracle; differs from how it was trained) | epoch 10, best_epoch 10, sha256 11912f0ec31f |
| mapper_tempo | (b) tempo | distorted | tempo model; embedding interpolated to the DISTORTED mel frame count, as in training | epoch 18, best_epoch 18, sha256 558bf8c929a2 |

## main: STOI/PESQ vs untouched clean, mel-domain inverse warp

| system | severity | stoi_melwarp | pesq_melwarp |
|---|---|---|---|
| clean (reference: untouched clean audio (ASR only)) | mild | – | – |
| clean (reference: untouched clean audio (ASR only)) | moderate | – | – |
| clean (reference: untouched clean audio (ASR only)) | severe | – | – |
| clean (reference: untouched clean audio (ASR only)) | all | – | – |
| distorted (floor) | mild | – | – |
| distorted (floor) | moderate | – | – |
| distorted (floor) | severe | – | – |
| distorted (floor) | all | – | – |
| pv_roundtrip (control: inverse warp only) | mild | – | – |
| pv_roundtrip (control: inverse warp only) | moderate | – | – |
| pv_roundtrip (control: inverse warp only) | severe | – | – |
| pv_roundtrip (control: inverse warp only) | all | – | – |
| copysynth_clean (ceiling (clean timeline)) | mild | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_clean (ceiling (clean timeline)) | moderate | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_clean (ceiling (clean timeline)) | severe | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_clean (ceiling (clean timeline)) | all | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_tempo (ceiling (distorted timeline)) | mild | 0.935 [0.929, 0.940] | 2.810 [2.721, 2.903] |
| copysynth_tempo (ceiling (distorted timeline)) | moderate | 0.935 [0.929, 0.940] | 2.812 [2.720, 2.905] |
| copysynth_tempo (ceiling (distorted timeline)) | severe | 0.934 [0.929, 0.939] | 2.803 [2.708, 2.900] |
| copysynth_tempo (ceiling (distorted timeline)) | all | 0.935 [0.929, 0.940] | 2.808 [2.717, 2.902] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | mild | 0.348 [0.338, 0.358] | 1.074 [1.066, 1.082] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | moderate | 0.347 [0.334, 0.361] | 1.079 [1.069, 1.091] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | severe | 0.327 [0.314, 0.339] | 1.076 [1.068, 1.085] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | all | 0.341 [0.330, 0.351] | 1.076 [1.068, 1.085] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | mild | 0.374 [0.364, 0.384] | 1.073 [1.065, 1.081] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | moderate | 0.391 [0.379, 0.403] | 1.075 [1.067, 1.084] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | severe | 0.390 [0.378, 0.401] | 1.074 [1.065, 1.082] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | all | 0.385 [0.374, 0.395] | 1.074 [1.066, 1.082] |
| mapper_tempo ((b) tempo) | mild | 0.602 [0.590, 0.613] | 1.128 [1.112, 1.145] |
| mapper_tempo ((b) tempo) | moderate | 0.618 [0.607, 0.628] | 1.130 [1.112, 1.148] |
| mapper_tempo ((b) tempo) | severe | 0.616 [0.605, 0.627] | 1.130 [1.113, 1.149] |
| mapper_tempo ((b) tempo) | all | 0.612 [0.601, 0.622] | 1.129 [1.112, 1.147] |

## intelligibility: ASR CER (primary) / WER, corpus level (sum of edits / sum of reference length)

| system | severity | cer | wer |
|---|---|---|---|
| clean (reference: untouched clean audio (ASR only)) | mild | 0.057 [0.048, 0.068] | 0.158 [0.137, 0.181] |
| clean (reference: untouched clean audio (ASR only)) | moderate | 0.057 [0.048, 0.068] | 0.158 [0.137, 0.181] |
| clean (reference: untouched clean audio (ASR only)) | severe | 0.057 [0.048, 0.068] | 0.158 [0.137, 0.181] |
| clean (reference: untouched clean audio (ASR only)) | all | 0.057 [0.048, 0.068] | 0.158 [0.137, 0.181] |
| distorted (floor) | mild | 0.236 [0.199, 0.277] | 0.493 [0.430, 0.559] |
| distorted (floor) | moderate | 0.483 [0.427, 0.546] | 0.833 [0.752, 0.923] |
| distorted (floor) | severe | 0.656 [0.603, 0.714] | 1.021 [0.948, 1.109] |
| distorted (floor) | all | 0.459 [0.416, 0.501] | 0.782 [0.721, 0.845] |
| pv_roundtrip (control: inverse warp only) | mild | – | – |
| pv_roundtrip (control: inverse warp only) | moderate | – | – |
| pv_roundtrip (control: inverse warp only) | severe | – | – |
| pv_roundtrip (control: inverse warp only) | all | – | – |
| copysynth_clean (ceiling (clean timeline)) | mild | – | – |
| copysynth_clean (ceiling (clean timeline)) | moderate | – | – |
| copysynth_clean (ceiling (clean timeline)) | severe | – | – |
| copysynth_clean (ceiling (clean timeline)) | all | – | – |
| copysynth_tempo (ceiling (distorted timeline)) | mild | 0.084 [0.069, 0.101] | 0.199 [0.173, 0.227] |
| copysynth_tempo (ceiling (distorted timeline)) | moderate | 0.087 [0.074, 0.102] | 0.224 [0.197, 0.253] |
| copysynth_tempo (ceiling (distorted timeline)) | severe | 0.102 [0.087, 0.118] | 0.256 [0.221, 0.293] |
| copysynth_tempo (ceiling (distorted timeline)) | all | 0.091 [0.078, 0.106] | 0.226 [0.199, 0.255] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | mild | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | moderate | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | severe | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | all | – | – |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | mild | 1.010 [0.955, 1.076] | 1.080 [1.034, 1.134] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | moderate | 0.967 [0.933, 1.006] | 1.085 [1.037, 1.141] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | severe | 0.971 [0.928, 1.025] | 1.047 [1.018, 1.085] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | all | 0.983 [0.951, 1.020] | 1.071 [1.041, 1.105] |
| mapper_tempo ((b) tempo) | mild | 0.537 [0.495, 0.578] | 0.907 [0.847, 0.965] |
| mapper_tempo ((b) tempo) | moderate | 0.645 [0.606, 0.686] | 1.022 [0.973, 1.070] |
| mapper_tempo ((b) tempo) | severe | 0.729 [0.683, 0.778] | 1.114 [1.056, 1.178] |
| mapper_tempo ((b) tempo) | all | 0.637 [0.601, 0.673] | 1.015 [0.965, 1.063] |

## secondary: STOI/PESQ vs untouched clean, waveform inverse warp (phase vocoder)

| system | severity | stoi | pesq |
|---|---|---|---|
| clean (reference: untouched clean audio (ASR only)) | mild | – | – |
| clean (reference: untouched clean audio (ASR only)) | moderate | – | – |
| clean (reference: untouched clean audio (ASR only)) | severe | – | – |
| clean (reference: untouched clean audio (ASR only)) | all | – | – |
| distorted (floor) | mild | 0.698 [0.684, 0.710] | 1.157 [1.141, 1.175] |
| distorted (floor) | moderate | 0.552 [0.535, 0.568] | 1.079 [1.070, 1.087] |
| distorted (floor) | severe | 0.479 [0.464, 0.494] | 1.064 [1.057, 1.071] |
| distorted (floor) | all | 0.576 [0.561, 0.590] | 1.100 [1.090, 1.111] |
| pv_roundtrip (control: inverse warp only) | mild | 0.791 [0.784, 0.798] | 1.570 [1.523, 1.619] |
| pv_roundtrip (control: inverse warp only) | moderate | 0.739 [0.729, 0.748] | 1.458 [1.418, 1.500] |
| pv_roundtrip (control: inverse warp only) | severe | 0.721 [0.713, 0.729] | 1.424 [1.386, 1.465] |
| pv_roundtrip (control: inverse warp only) | all | 0.750 [0.743, 0.757] | 1.484 [1.443, 1.526] |
| copysynth_clean (ceiling (clean timeline)) | mild | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_clean (ceiling (clean timeline)) | moderate | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_clean (ceiling (clean timeline)) | severe | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_clean (ceiling (clean timeline)) | all | 0.942 [0.937, 0.946] | 2.932 [2.841, 3.025] |
| copysynth_tempo (ceiling (distorted timeline)) | mild | 0.755 [0.747, 0.763] | 1.557 [1.515, 1.600] |
| copysynth_tempo (ceiling (distorted timeline)) | moderate | 0.722 [0.711, 0.731] | 1.459 [1.419, 1.500] |
| copysynth_tempo (ceiling (distorted timeline)) | severe | 0.714 [0.704, 0.723] | 1.437 [1.403, 1.473] |
| copysynth_tempo (ceiling (distorted timeline)) | all | 0.730 [0.721, 0.738] | 1.484 [1.447, 1.523] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | mild | 0.348 [0.338, 0.358] | 1.074 [1.066, 1.082] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | moderate | 0.347 [0.334, 0.361] | 1.079 [1.069, 1.091] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | severe | 0.327 [0.314, 0.339] | 1.076 [1.068, 1.085] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | all | 0.341 [0.330, 0.351] | 1.076 [1.068, 1.085] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | mild | 0.365 [0.354, 0.375] | 1.071 [1.063, 1.080] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | moderate | 0.375 [0.363, 0.387] | 1.072 [1.064, 1.080] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | severe | 0.372 [0.360, 0.383] | 1.069 [1.062, 1.077] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | all | 0.371 [0.360, 0.381] | 1.071 [1.063, 1.079] |
| mapper_tempo ((b) tempo) | mild | 0.575 [0.565, 0.584] | 1.115 [1.100, 1.131] |
| mapper_tempo ((b) tempo) | moderate | 0.576 [0.565, 0.586] | 1.112 [1.097, 1.127] |
| mapper_tempo ((b) tempo) | severe | 0.570 [0.560, 0.581] | 1.111 [1.097, 1.126] |
| mapper_tempo ((b) tempo) | all | 0.573 [0.563, 0.583] | 1.113 [1.098, 1.128] |

## mel L1

| system | severity | mel_l1 | mel_l1_clean |
|---|---|---|---|
| clean (reference: untouched clean audio (ASR only)) | mild | – | – |
| clean (reference: untouched clean audio (ASR only)) | moderate | – | – |
| clean (reference: untouched clean audio (ASR only)) | severe | – | – |
| clean (reference: untouched clean audio (ASR only)) | all | – | – |
| distorted (floor) | mild | 1.977 [1.801, 2.171] | 1.985 [1.808, 2.179] |
| distorted (floor) | moderate | 2.386 [2.209, 2.579] | 2.394 [2.217, 2.588] |
| distorted (floor) | severe | 2.960 [2.791, 3.143] | 2.975 [2.803, 3.161] |
| distorted (floor) | all | 2.441 [2.267, 2.630] | 2.451 [2.276, 2.642] |
| pv_roundtrip (control: inverse warp only) | mild | 0.450 [0.433, 0.466] | 0.453 [0.436, 0.470] |
| pv_roundtrip (control: inverse warp only) | moderate | 0.507 [0.488, 0.526] | 0.514 [0.494, 0.534] |
| pv_roundtrip (control: inverse warp only) | severe | 0.524 [0.505, 0.543] | 0.534 [0.514, 0.553] |
| pv_roundtrip (control: inverse warp only) | all | 0.494 [0.476, 0.511] | 0.500 [0.482, 0.519] |
| copysynth_clean (ceiling (clean timeline)) | mild | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| copysynth_clean (ceiling (clean timeline)) | moderate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| copysynth_clean (ceiling (clean timeline)) | severe | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| copysynth_clean (ceiling (clean timeline)) | all | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| copysynth_tempo (ceiling (distorted timeline)) | mild | 0.000 [0.000, 0.000] | 0.079 [0.076, 0.082] |
| copysynth_tempo (ceiling (distorted timeline)) | moderate | 0.000 [0.000, 0.000] | 0.079 [0.076, 0.082] |
| copysynth_tempo (ceiling (distorted timeline)) | severe | 0.000 [0.000, 0.000] | 0.079 [0.076, 0.082] |
| copysynth_tempo (ceiling (distorted timeline)) | all | 0.000 [0.000, 0.000] | 0.079 [0.076, 0.082] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | mild | 1.326 [1.244, 1.414] | 1.326 [1.244, 1.414] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | moderate | 1.353 [1.271, 1.442] | 1.353 [1.271, 1.442] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | severe | 1.395 [1.304, 1.497] | 1.395 [1.304, 1.497] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | all | 1.358 [1.274, 1.450] | 1.358 [1.274, 1.450] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | mild | 1.291 [1.210, 1.378] | 1.306 [1.225, 1.393] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | moderate | 1.288 [1.206, 1.378] | 1.304 [1.223, 1.394] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | severe | 1.308 [1.221, 1.405] | 1.328 [1.241, 1.425] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | all | 1.295 [1.213, 1.386] | 1.313 [1.230, 1.403] |
| mapper_tempo ((b) tempo) | mild | 1.096 [1.025, 1.179] | 1.107 [1.037, 1.190] |
| mapper_tempo ((b) tempo) | moderate | 1.082 [1.013, 1.162] | 1.095 [1.027, 1.174] |
| mapper_tempo ((b) tempo) | severe | 1.104 [1.031, 1.188] | 1.122 [1.049, 1.205] |
| mapper_tempo ((b) tempo) | all | 1.094 [1.024, 1.175] | 1.108 [1.039, 1.188] |

## speaker similarity (ECAPA cosine)

| system | severity | spk_sim_same | spk_sim_other |
|---|---|---|---|
| clean (reference: untouched clean audio (ASR only)) | mild | – | – |
| clean (reference: untouched clean audio (ASR only)) | moderate | – | – |
| clean (reference: untouched clean audio (ASR only)) | severe | – | – |
| clean (reference: untouched clean audio (ASR only)) | all | – | – |
| distorted (floor) | mild | 0.645 [0.629, 0.660] | 0.429 [0.409, 0.450] |
| distorted (floor) | moderate | 0.447 [0.430, 0.463] | 0.315 [0.298, 0.332] |
| distorted (floor) | severe | 0.280 [0.267, 0.291] | 0.206 [0.194, 0.218] |
| distorted (floor) | all | 0.457 [0.445, 0.469] | 0.317 [0.302, 0.332] |
| pv_roundtrip (control: inverse warp only) | mild | 0.894 [0.886, 0.901] | 0.545 [0.522, 0.569] |
| pv_roundtrip (control: inverse warp only) | moderate | 0.842 [0.831, 0.851] | 0.523 [0.499, 0.545] |
| pv_roundtrip (control: inverse warp only) | severe | 0.822 [0.809, 0.834] | 0.513 [0.488, 0.536] |
| pv_roundtrip (control: inverse warp only) | all | 0.852 [0.843, 0.861] | 0.527 [0.503, 0.550] |
| copysynth_clean (ceiling (clean timeline)) | mild | 0.948 [0.945, 0.951] | 0.568 [0.544, 0.591] |
| copysynth_clean (ceiling (clean timeline)) | moderate | 0.948 [0.945, 0.951] | 0.568 [0.544, 0.591] |
| copysynth_clean (ceiling (clean timeline)) | severe | 0.948 [0.945, 0.951] | 0.568 [0.544, 0.591] |
| copysynth_clean (ceiling (clean timeline)) | all | 0.948 [0.945, 0.951] | 0.568 [0.544, 0.591] |
| copysynth_tempo (ceiling (distorted timeline)) | mild | 0.938 [0.933, 0.942] | 0.563 [0.539, 0.587] |
| copysynth_tempo (ceiling (distorted timeline)) | moderate | 0.926 [0.921, 0.931] | 0.558 [0.534, 0.582] |
| copysynth_tempo (ceiling (distorted timeline)) | severe | 0.916 [0.910, 0.921] | 0.556 [0.532, 0.580] |
| copysynth_tempo (ceiling (distorted timeline)) | all | 0.927 [0.922, 0.931] | 0.559 [0.535, 0.582] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | mild | 0.077 [0.064, 0.089] | 0.072 [0.060, 0.084] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | moderate | 0.078 [0.066, 0.091] | 0.073 [0.061, 0.085] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | severe | 0.081 [0.070, 0.094] | 0.074 [0.062, 0.086] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | all | 0.079 [0.067, 0.091] | 0.073 [0.061, 0.085] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | mild | 0.077 [0.065, 0.089] | 0.072 [0.060, 0.083] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | moderate | 0.077 [0.065, 0.089] | 0.072 [0.060, 0.084] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | severe | 0.080 [0.067, 0.093] | 0.073 [0.061, 0.086] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | all | 0.078 [0.066, 0.090] | 0.072 [0.060, 0.084] |
| mapper_tempo ((b) tempo) | mild | 0.243 [0.226, 0.260] | 0.202 [0.185, 0.218] |
| mapper_tempo ((b) tempo) | moderate | 0.245 [0.228, 0.262] | 0.206 [0.189, 0.222] |
| mapper_tempo ((b) tempo) | severe | 0.237 [0.221, 0.252] | 0.200 [0.185, 0.215] |
| mapper_tempo ((b) tempo) | all | 0.242 [0.226, 0.257] | 0.203 [0.187, 0.218] |

Clean references for spk_sim_other: clean: other clip, same speaker 0.583 [0.559, 0.607]; clean: clips of other speakers 0.200 [0.187, 0.213]

## pitch / tone vs clean (Praat; melwarp, or waveform inverse warp for the floor)

| system | severity | f0_corr | f0_rmse_st | vuv_agree | hnr_db |
|---|---|---|---|---|---|
| clean (reference: untouched clean audio (ASR only)) | mild | – | – | – | – |
| clean (reference: untouched clean audio (ASR only)) | moderate | – | – | – | – |
| clean (reference: untouched clean audio (ASR only)) | severe | – | – | – | – |
| clean (reference: untouched clean audio (ASR only)) | all | – | – | – | – |
| distorted (floor) | mild | 0.873 [0.847, 0.895] | 1.660 [1.534, 1.805] | 0.914 [0.908, 0.921] | 9.654 [9.020, 10.243] |
| distorted (floor) | moderate | 0.812 [0.777, 0.845] | 2.767 [2.609, 2.951] | 0.884 [0.875, 0.893] | 7.692 [7.079, 8.290] |
| distorted (floor) | severe | 0.706 [0.652, 0.754] | 3.971 [3.722, 4.271] | 0.847 [0.833, 0.859] | 4.345 [3.908, 4.782] |
| distorted (floor) | all | 0.797 [0.764, 0.826] | 2.798 [2.641, 2.986] | 0.882 [0.873, 0.890] | 7.230 [6.687, 7.754] |
| pv_roundtrip (control: inverse warp only) | mild | – | – | – | – |
| pv_roundtrip (control: inverse warp only) | moderate | – | – | – | – |
| pv_roundtrip (control: inverse warp only) | severe | – | – | – | – |
| pv_roundtrip (control: inverse warp only) | all | – | – | – | – |
| copysynth_clean (ceiling (clean timeline)) | mild | 0.885 [0.853, 0.912] | 1.162 [0.966, 1.378] | 0.967 [0.960, 0.974] | 12.832 [12.001, 13.626] |
| copysynth_clean (ceiling (clean timeline)) | moderate | 0.885 [0.853, 0.912] | 1.162 [0.966, 1.378] | 0.967 [0.960, 0.974] | 12.832 [12.001, 13.626] |
| copysynth_clean (ceiling (clean timeline)) | severe | 0.885 [0.853, 0.912] | 1.162 [0.966, 1.378] | 0.967 [0.960, 0.974] | 12.832 [12.001, 13.626] |
| copysynth_clean (ceiling (clean timeline)) | all | 0.885 [0.853, 0.912] | 1.162 [0.966, 1.378] | 0.967 [0.960, 0.974] | 12.832 [12.001, 13.626] |
| copysynth_tempo (ceiling (distorted timeline)) | mild | 0.886 [0.861, 0.910] | 1.205 [1.018, 1.406] | 0.967 [0.960, 0.973] | 13.154 [12.349, 13.921] |
| copysynth_tempo (ceiling (distorted timeline)) | moderate | 0.890 [0.861, 0.915] | 1.171 [0.990, 1.367] | 0.967 [0.960, 0.973] | 13.177 [12.378, 13.939] |
| copysynth_tempo (ceiling (distorted timeline)) | severe | 0.894 [0.865, 0.919] | 1.151 [0.963, 1.360] | 0.967 [0.960, 0.973] | 13.114 [12.316, 13.879] |
| copysynth_tempo (ceiling (distorted timeline)) | all | 0.890 [0.863, 0.913] | 1.176 [0.995, 1.373] | 0.967 [0.960, 0.973] | 13.148 [12.348, 13.912] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | mild | – | – | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | moderate | – | – | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | severe | – | – | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | all | – | – | – | – |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | mild | – | – | – | – |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | moderate | – | – | – | – |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | severe | – | – | – | – |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | all | – | – | – | – |
| mapper_tempo ((b) tempo) | mild | 0.431 [0.364, 0.491] | 5.209 [4.902, 5.524] | 0.817 [0.795, 0.838] | 4.608 [4.029, 5.180] |
| mapper_tempo ((b) tempo) | moderate | 0.470 [0.402, 0.533] | 4.913 [4.495, 5.334] | 0.825 [0.800, 0.849] | 4.981 [4.294, 5.662] |
| mapper_tempo ((b) tempo) | severe | 0.418 [0.343, 0.486] | 5.084 [4.709, 5.461] | 0.825 [0.801, 0.847] | 4.949 [4.264, 5.625] |
| mapper_tempo ((b) tempo) | all | 0.440 [0.373, 0.500] | 5.069 [4.733, 5.408] | 0.822 [0.799, 0.844] | 4.846 [4.205, 5.482] |

Clean reference HNR (dB): 13.446 [12.633, 14.227]

Rows with < 10 frames voiced in both (f0_corr / f0_rmse_st NaN): distorted 1, copysynth_clean 0, copysynth_tempo 0, mapper_tempo 4

## legacy

| system | severity | stoi_trunc | pesq_trunc | stoi_fwd | pesq_fwd |
|---|---|---|---|---|---|
| clean (reference: untouched clean audio (ASR only)) | mild | – | – | – | – |
| clean (reference: untouched clean audio (ASR only)) | moderate | – | – | – | – |
| clean (reference: untouched clean audio (ASR only)) | severe | – | – | – | – |
| clean (reference: untouched clean audio (ASR only)) | all | – | – | – | – |
| distorted (floor) | mild | 0.393 [0.369, 0.415] | 1.165 [1.148, 1.182] | 0.801 [0.787, 0.814] | 1.256 [1.229, 1.284] |
| distorted (floor) | moderate | 0.261 [0.243, 0.281] | 1.081 [1.072, 1.089] | 0.616 [0.600, 0.632] | 1.104 [1.093, 1.116] |
| distorted (floor) | severe | 0.245 [0.232, 0.257] | 1.067 [1.060, 1.075] | 0.510 [0.495, 0.524] | 1.082 [1.073, 1.092] |
| distorted (floor) | all | 0.300 [0.286, 0.313] | 1.104 [1.094, 1.115] | 0.642 [0.628, 0.656] | 1.147 [1.132, 1.163] |
| pv_roundtrip (control: inverse warp only) | mild | 0.402 [0.376, 0.427] | 1.680 [1.631, 1.730] | 1.000 [1.000, 1.000] | 4.644 [4.644, 4.644] |
| pv_roundtrip (control: inverse warp only) | moderate | 0.245 [0.226, 0.266] | 1.448 [1.406, 1.491] | 1.000 [1.000, 1.000] | 4.644 [4.644, 4.644] |
| pv_roundtrip (control: inverse warp only) | severe | 0.208 [0.194, 0.222] | 1.359 [1.327, 1.393] | 1.000 [1.000, 1.000] | 4.644 [4.644, 4.644] |
| pv_roundtrip (control: inverse warp only) | all | 0.285 [0.272, 0.298] | 1.495 [1.458, 1.535] | 1.000 [1.000, 1.000] | 4.644 [4.644, 4.644] |
| copysynth_clean (ceiling (clean timeline)) | mild | 0.942 [0.937, 0.946] | 2.926 [2.836, 3.019] | – | – |
| copysynth_clean (ceiling (clean timeline)) | moderate | 0.942 [0.937, 0.946] | 2.926 [2.836, 3.019] | – | – |
| copysynth_clean (ceiling (clean timeline)) | severe | 0.942 [0.937, 0.946] | 2.926 [2.836, 3.019] | – | – |
| copysynth_clean (ceiling (clean timeline)) | all | 0.942 [0.937, 0.946] | 2.926 [2.836, 3.019] | – | – |
| copysynth_tempo (ceiling (distorted timeline)) | mild | 0.397 [0.369, 0.425] | 1.831 [1.776, 1.889] | 0.751 [0.740, 0.761] | 1.604 [1.558, 1.651] |
| copysynth_tempo (ceiling (distorted timeline)) | moderate | 0.230 [0.211, 0.250] | 1.498 [1.459, 1.539] | 0.709 [0.699, 0.720] | 1.494 [1.453, 1.537] |
| copysynth_tempo (ceiling (distorted timeline)) | severe | 0.193 [0.179, 0.206] | 1.383 [1.350, 1.417] | 0.698 [0.688, 0.708] | 1.464 [1.426, 1.503] |
| copysynth_tempo (ceiling (distorted timeline)) | all | 0.273 [0.261, 0.286] | 1.571 [1.533, 1.610] | 0.720 [0.710, 0.729] | 1.521 [1.481, 1.561] |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | mild | 0.348 [0.338, 0.358] | 1.074 [1.066, 1.082] | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | moderate | 0.347 [0.334, 0.361] | 1.074 [1.066, 1.083] | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | severe | 0.327 [0.314, 0.339] | 1.079 [1.070, 1.089] | – | – |
| mapper_linear_oracle ((a) linear, clean length (oracle)) | all | 0.341 [0.330, 0.351] | 1.076 [1.068, 1.084] | – | – |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | mild | 0.333 [0.322, 0.343] | 1.073 [1.065, 1.080] | 0.326 [0.316, 0.336] | 1.083 [1.074, 1.093] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | moderate | 0.312 [0.298, 0.327] | 1.076 [1.067, 1.085] | 0.323 [0.312, 0.334] | 1.087 [1.078, 1.097] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | severe | 0.302 [0.288, 0.316] | 1.074 [1.066, 1.084] | 0.305 [0.294, 0.315] | 1.088 [1.078, 1.098] |
| mapper_linear_deploy ((a) linear, distorted length (deployable)) | all | 0.316 [0.304, 0.327] | 1.074 [1.066, 1.083] | 0.318 [0.308, 0.327] | 1.086 [1.077, 1.096] |
| mapper_tempo ((b) tempo) | mild | 0.353 [0.332, 0.373] | 1.121 [1.107, 1.136] | 0.571 [0.560, 0.581] | 1.141 [1.124, 1.159] |
| mapper_tempo ((b) tempo) | moderate | 0.233 [0.216, 0.251] | 1.114 [1.101, 1.129] | 0.595 [0.584, 0.607] | 1.149 [1.131, 1.168] |
| mapper_tempo ((b) tempo) | severe | 0.204 [0.192, 0.216] | 1.108 [1.095, 1.122] | 0.594 [0.582, 0.605] | 1.149 [1.132, 1.168] |
| mapper_tempo ((b) tempo) | all | 0.263 [0.253, 0.274] | 1.115 [1.101, 1.128] | 0.587 [0.576, 0.597] | 1.146 [1.129, 1.165] |

## Paired: mapper_tempo − mapper_linear_oracle (per clip)

| severity | metric | better | mean diff [95% CI] | share mapper_tempo better | p (Wilcoxon, speakers) | p (Wilcoxon, clips) |
|---|---|---|---|---|---|---|
| mild | stoi_melwarp | higher | 0.2535 [0.2409, 0.2666] | 1.00 | 7.11e-15 | 2.99e-52 |
| mild | pesq_melwarp | higher | 0.0544 [0.0444, 0.0650] | 0.97 | 7.11e-15 | 6.07e-48 |
| mild | stoi | higher | 0.2266 [0.2155, 0.2382] | 1.00 | 7.11e-15 | 3.02e-52 |
| mild | pesq | higher | 0.0414 [0.0329, 0.0508] | 0.96 | 3.55e-14 | 1.27e-46 |
| mild | mel_l1_clean | lower | -0.2186 [-0.2699, -0.1678] | 0.81 | 1.24e-11 | 4.36e-34 |
| mild | spk_sim_same | higher | 0.1663 [0.1482, 0.1844] | 0.96 | 7.11e-15 | 5.99e-51 |
| mild | spk_sim_other | higher | 0.1300 [0.1130, 0.1473] | 0.95 | 7.11e-15 | 1.61e-50 |
| mild | stoi_trunc | higher | 0.0048 [-0.0151, 0.0242] | 0.53 | 0.733 | 0.56 |
| mild | pesq_trunc | higher | 0.0469 [0.0387, 0.0556] | 0.95 | 1.42e-14 | 2.44e-46 |
| moderate | stoi_melwarp | higher | 0.2704 [0.2581, 0.2835] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | pesq_melwarp | higher | 0.0504 [0.0371, 0.0634] | 0.95 | 2.86e-10 | 7.84e-45 |
| moderate | stoi | higher | 0.2282 [0.2168, 0.2402] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | pesq | higher | 0.0326 [0.0212, 0.0433] | 0.91 | 4.3e-09 | 2.67e-40 |
| moderate | mel_l1_clean | lower | -0.2573 [-0.3043, -0.2100] | 0.87 | 1.78e-13 | 5.78e-39 |
| moderate | spk_sim_same | higher | 0.1667 [0.1506, 0.1824] | 0.97 | 7.11e-15 | 1.59e-51 |
| moderate | spk_sim_other | higher | 0.1330 [0.1167, 0.1494] | 0.95 | 7.11e-15 | 6.17e-51 |
| moderate | stoi_trunc | higher | -0.1143 [-0.1279, -0.1003] | 0.22 | 7.11e-14 | 9.69e-30 |
| moderate | pesq_trunc | higher | 0.0401 [0.0332, 0.0475] | 0.92 | 9.95e-14 | 1.24e-44 |
| severe | stoi_melwarp | higher | 0.2893 [0.2773, 0.3022] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | pesq_melwarp | higher | 0.0537 [0.0422, 0.0661] | 0.92 | 3.06e-13 | 3.5e-43 |
| severe | stoi | higher | 0.2436 [0.2318, 0.2561] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | pesq | higher | 0.0347 [0.0260, 0.0442] | 0.89 | 8.09e-11 | 1.75e-36 |
| severe | mel_l1_clean | lower | -0.2732 [-0.3181, -0.2290] | 0.89 | 7.11e-15 | 3.68e-44 |
| severe | spk_sim_same | higher | 0.1555 [0.1414, 0.1694] | 0.96 | 7.11e-15 | 4.99e-51 |
| severe | spk_sim_other | higher | 0.1263 [0.1126, 0.1401] | 0.97 | 7.11e-15 | 8.73e-52 |
| severe | stoi_trunc | higher | -0.1223 [-0.1340, -0.1112] | 0.13 | 7.11e-15 | 7.18e-40 |
| severe | pesq_trunc | higher | 0.0291 [0.0215, 0.0366] | 0.86 | 1.76e-08 | 4.25e-35 |
| all | stoi_melwarp | higher | 0.2710 [0.2600, 0.2823] | 1.00 | 7.11e-15 | 8.25e-153 |
| all | pesq_melwarp | higher | 0.0528 [0.0423, 0.0641] | 0.94 | 2.34e-13 | 4.74e-132 |
| all | stoi | higher | 0.2328 [0.2227, 0.2432] | 1.00 | 7.11e-15 | 8.28e-153 |
| all | pesq | higher | 0.0362 [0.0279, 0.0451] | 0.92 | 1.69e-11 | 1.38e-119 |
| all | mel_l1_clean | lower | -0.2497 [-0.2942, -0.2048] | 0.86 | 7.11e-15 | 2.19e-113 |
| all | spk_sim_same | higher | 0.1628 [0.1473, 0.1781] | 0.96 | 7.11e-15 | 1.22e-149 |
| all | spk_sim_other | higher | 0.1298 [0.1146, 0.1452] | 0.96 | 7.11e-15 | 1.55e-149 |
| all | stoi_trunc | higher | -0.0773 [-0.0860, -0.0687] | 0.30 | 7.11e-15 | 9.21e-43 |
| all | pesq_trunc | higher | 0.0387 [0.0320, 0.0456] | 0.91 | 1.42e-14 | 6.47e-123 |

## Paired: mapper_tempo − mapper_linear_deploy (per clip)

| severity | metric | better | mean diff [95% CI] | share mapper_tempo better | p (Wilcoxon, speakers) | p (Wilcoxon, clips) |
|---|---|---|---|---|---|---|
| mild | stoi_melwarp | higher | 0.2277 [0.2161, 0.2394] | 1.00 | 7.11e-15 | 2.99e-52 |
| mild | pesq_melwarp | higher | 0.0548 [0.0452, 0.0651] | 0.97 | 7.11e-15 | 4.49e-49 |
| mild | cer | lower | -0.4732 [-0.5486, -0.4028] | 0.93 | 7.11e-15 | 3.04e-47 |
| mild | wer | lower | -0.1727 [-0.2515, -0.0992] | 0.51 | 0.000139 | 6.07e-05 |
| mild | stoi | higher | 0.2100 [0.2001, 0.2201] | 1.00 | 7.11e-15 | 2.99e-52 |
| mild | pesq | higher | 0.0438 [0.0358, 0.0526] | 0.97 | 7.11e-15 | 1.79e-51 |
| mild | mel_l1_clean | lower | -0.1982 [-0.2484, -0.1488] | 0.81 | 9.03e-10 | 1.91e-31 |
| mild | spk_sim_same | higher | 0.1664 [0.1484, 0.1841] | 0.96 | 7.11e-15 | 3.66e-51 |
| mild | spk_sim_other | higher | 0.1303 [0.1136, 0.1473] | 0.95 | 7.11e-15 | 8.24e-51 |
| mild | stoi_trunc | higher | 0.0204 [0.0009, 0.0392] | 0.55 | 0.102 | 0.023 |
| mild | pesq_trunc | higher | 0.0484 [0.0403, 0.0570] | 0.95 | 7.11e-15 | 9.83e-49 |
| moderate | stoi_melwarp | higher | 0.2264 [0.2158, 0.2376] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | pesq_melwarp | higher | 0.0542 [0.0434, 0.0658] | 0.95 | 1.42e-14 | 5.47e-48 |
| moderate | cer | lower | -0.3216 [-0.3852, -0.2626] | 0.88 | 2.64e-12 | 4.83e-44 |
| moderate | wer | lower | -0.0632 [-0.1468, 0.0141] | 0.40 | 0.593 | 0.928 |
| moderate | stoi | higher | 0.2001 [0.1902, 0.2108] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | pesq | higher | 0.0398 [0.0314, 0.0489] | 0.94 | 3.55e-14 | 1.63e-46 |
| moderate | mel_l1_clean | lower | -0.2090 [-0.2548, -0.1622] | 0.83 | 1.45e-11 | 7.84e-34 |
| moderate | spk_sim_same | higher | 0.1683 [0.1523, 0.1840] | 0.96 | 7.11e-15 | 1.48e-51 |
| moderate | spk_sim_other | higher | 0.1340 [0.1173, 0.1505] | 0.96 | 7.11e-15 | 6.35e-51 |
| moderate | stoi_trunc | higher | -0.0793 [-0.0890, -0.0692] | 0.24 | 1.35e-13 | 2.32e-24 |
| moderate | pesq_trunc | higher | 0.0386 [0.0305, 0.0472] | 0.91 | 1.05e-10 | 3.55e-42 |
| severe | stoi_melwarp | higher | 0.2263 [0.2142, 0.2390] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | pesq_melwarp | higher | 0.0567 [0.0462, 0.0680] | 0.97 | 7.11e-15 | 7.31e-48 |
| severe | cer | lower | -0.2423 [-0.3144, -0.1726] | 0.83 | 1.25e-09 | 1.36e-33 |
| severe | wer | lower | 0.0675 [0.0041, 0.1354] | 0.32 | 0.0478 | 0.00123 |
| severe | stoi | higher | 0.1985 [0.1871, 0.2106] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | pesq | higher | 0.0419 [0.0339, 0.0507] | 0.95 | 7.11e-15 | 5.09e-50 |
| severe | mel_l1_clean | lower | -0.2057 [-0.2479, -0.1633] | 0.85 | 3.06e-13 | 3.01e-36 |
| severe | spk_sim_same | higher | 0.1570 [0.1416, 0.1718] | 0.97 | 7.11e-15 | 5.82e-51 |
| severe | spk_sim_other | higher | 0.1271 [0.1129, 0.1416] | 0.97 | 7.11e-15 | 1.39e-51 |
| severe | stoi_trunc | higher | -0.0980 [-0.1093, -0.0870] | 0.18 | 4.97e-14 | 4.49e-37 |
| severe | pesq_trunc | higher | 0.0339 [0.0267, 0.0413] | 0.92 | 1.55e-10 | 1.4e-43 |
| all | stoi_melwarp | higher | 0.2268 [0.2163, 0.2378] | 1.00 | 7.11e-15 | 8.25e-153 |
| all | pesq_melwarp | higher | 0.0552 [0.0452, 0.0660] | 0.96 | 7.11e-15 | 4.65e-141 |
| all | cer | lower | -0.3457 [-0.3985, -0.2942] | 0.88 | 3.55e-14 | 8.68e-122 |
| all | wer | lower | -0.0561 [-0.1165, 0.0012] | 0.41 | 0.1 | 0.555 |
| all | stoi | higher | 0.2029 [0.1932, 0.2131] | 1.00 | 7.11e-15 | 8.25e-153 |
| all | pesq | higher | 0.0418 [0.0339, 0.0503] | 0.96 | 7.11e-15 | 4.65e-144 |
| all | mel_l1_clean | lower | -0.2043 [-0.2477, -0.1598] | 0.83 | 1.47e-12 | 1.25e-97 |
| all | spk_sim_same | higher | 0.1639 [0.1481, 0.1793] | 0.96 | 7.11e-15 | 8.19e-150 |
| all | spk_sim_other | higher | 0.1305 [0.1151, 0.1460] | 0.96 | 7.11e-15 | 1.39e-149 |
| all | stoi_trunc | higher | -0.0523 [-0.0609, -0.0439] | 0.32 | 4.97e-13 | 1.65e-30 |
| all | pesq_trunc | higher | 0.0403 [0.0332, 0.0478] | 0.93 | 7.82e-13 | 4.03e-131 |

## Paired: mapper_tempo − distorted (per clip)

| severity | metric | better | mean diff [95% CI] | share mapper_tempo better | p (Wilcoxon, speakers) | p (Wilcoxon, clips) |
|---|---|---|---|---|---|---|
| mild | cer | lower | 0.3005 [0.2721, 0.3300] | 0.08 | 1.42e-14 | 2.05e-42 |
| mild | wer | lower | 0.4144 [0.3698, 0.4589] | 0.12 | 7.11e-15 | 4.98e-32 |
| mild | stoi | higher | -0.1230 [-0.1315, -0.1139] | 0.01 | 7.11e-15 | 4.13e-52 |
| mild | pesq | higher | -0.0425 [-0.0520, -0.0328] | 0.20 | 1.36e-10 | 4.85e-31 |
| mild | mel_l1_clean | lower | -0.8776 [-1.0233, -0.7410] | 0.94 | 1.78e-13 | 2.22e-50 |
| mild | spk_sim_same | higher | -0.4021 [-0.4229, -0.3817] | 0.01 | 7.11e-15 | 3.08e-52 |
| mild | spk_sim_other | higher | -0.2271 [-0.2432, -0.2112] | 0.01 | 7.11e-15 | 3.3e-52 |
| mild | f0_corr | higher | -0.4411 [-0.5053, -0.3837] | 0.04 | 7.11e-15 | 2.68e-49 |
| mild | f0_rmse_st | lower | 3.5448 [3.2492, 3.8384] | 0.03 | 7.11e-15 | 8.89e-51 |
| mild | vuv_agree | higher | -0.0977 [-0.1169, -0.0791] | 0.10 | 7.11e-15 | 1.07e-45 |
| mild | hnr_db | higher | -5.0456 [-5.3332, -4.7500] | 0.01 | 7.11e-15 | 3.14e-52 |
| mild | stoi_trunc | higher | -0.0396 [-0.0505, -0.0289] | 0.33 | 1.76e-08 | 1.46e-13 |
| mild | pesq_trunc | higher | -0.0437 [-0.0522, -0.0357] | 0.22 | 6.42e-12 | 1.11e-28 |
| moderate | cer | lower | 0.1618 [0.1103, 0.2119] | 0.23 | 6.57e-07 | 3.28e-22 |
| moderate | wer | lower | 0.1890 [0.1121, 0.2586] | 0.25 | 2.92e-06 | 3.9e-13 |
| moderate | stoi | higher | 0.0237 [0.0147, 0.0329] | 0.72 | 3.76e-06 | 4.7e-17 |
| moderate | pesq | higher | 0.0333 [0.0251, 0.0420] | 0.83 | 3.91e-13 | 8.66e-36 |
| moderate | mel_l1_clean | lower | -1.2989 [-1.4385, -1.1647] | 0.99 | 7.11e-15 | 3.17e-52 |
| moderate | spk_sim_same | higher | -0.2024 [-0.2252, -0.1784] | 0.06 | 1.42e-14 | 4.1e-48 |
| moderate | spk_sim_other | higher | -0.1097 [-0.1249, -0.0930] | 0.11 | 3.06e-13 | 1.13e-43 |
| moderate | f0_corr | higher | -0.3427 [-0.3969, -0.2905] | 0.09 | 1.42e-14 | 3.32e-39 |
| moderate | f0_rmse_st | lower | 2.1467 [1.7640, 2.5376] | 0.20 | 1.35e-13 | 3.8e-35 |
| moderate | vuv_agree | higher | -0.0589 [-0.0792, -0.0403] | 0.23 | 1.91e-09 | 1.82e-25 |
| moderate | hnr_db | higher | -2.7118 [-2.9846, -2.4330] | 0.08 | 7.11e-15 | 4.58e-48 |
| moderate | stoi_trunc | higher | -0.0283 [-0.0352, -0.0213] | 0.31 | 8.81e-08 | 4.12e-14 |
| moderate | pesq_trunc | higher | 0.0340 [0.0272, 0.0411] | 0.83 | 2.13e-14 | 9.98e-34 |
| severe | cer | lower | 0.0732 [0.0058, 0.1348] | 0.38 | 0.00663 | 0.00019 |
| severe | wer | lower | 0.0933 [0.0015, 0.1746] | 0.31 | 0.0131 | 0.00288 |
| severe | stoi | higher | 0.0907 [0.0832, 0.0983] | 0.98 | 7.11e-15 | 4.68e-52 |
| severe | pesq | higher | 0.0472 [0.0387, 0.0564] | 0.96 | 7.11e-15 | 9.99e-51 |
| severe | mel_l1_clean | lower | -1.8532 [-1.9775, -1.7313] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | spk_sim_same | higher | -0.0429 [-0.0619, -0.0234] | 0.34 | 3.97e-05 | 3.24e-10 |
| severe | spk_sim_other | higher | -0.0064 [-0.0208, 0.0085] | 0.45 | 0.346 | 0.118 |
| severe | f0_corr | higher | -0.2864 [-0.3462, -0.2303] | 0.16 | 1.8e-12 | 1.7e-30 |
| severe | f0_rmse_st | lower | 1.0992 [0.7130, 1.4867] | 0.34 | 1.63e-06 | 1.59e-13 |
| severe | vuv_agree | higher | -0.0220 [-0.0366, -0.0086] | 0.42 | 0.0477 | 5.47e-06 |
| severe | hnr_db | higher | 0.6034 [0.2595, 0.9347] | 0.61 | 0.000307 | 1.26e-07 |
| severe | stoi_trunc | higher | -0.0405 [-0.0478, -0.0331] | 0.24 | 2.18e-12 | 3.92e-24 |
| severe | pesq_trunc | higher | 0.0410 [0.0331, 0.0496] | 0.91 | 7.11e-15 | 1.57e-44 |
| all | cer | lower | 0.1785 [0.1450, 0.2109] | 0.23 | 5.77e-10 | 4.59e-61 |
| all | wer | lower | 0.2322 [0.1841, 0.2778] | 0.23 | 1.22e-08 | 1.47e-41 |
| all | stoi | higher | -0.0029 [-0.0105, 0.0050] | 0.57 | 0.673 | 0.832 |
| all | pesq | higher | 0.0127 [0.0056, 0.0203] | 0.67 | 0.00254 | 2.49e-16 |
| all | mel_l1_clean | lower | -1.3432 [-1.4775, -1.2138] | 0.98 | 7.11e-15 | 5.6e-152 |
| all | spk_sim_same | higher | -0.2158 [-0.2350, -0.1963] | 0.14 | 7.11e-15 | 1.39e-120 |
| all | spk_sim_other | higher | -0.1144 [-0.1276, -0.1003] | 0.19 | 1.42e-14 | 6.81e-98 |
| all | f0_corr | higher | -0.3568 [-0.4084, -0.3109] | 0.10 | 7.11e-15 | 1.45e-115 |
| all | f0_rmse_st | lower | 2.2646 [1.9583, 2.5776] | 0.19 | 7.11e-15 | 3.22e-98 |
| all | vuv_agree | higher | -0.0595 [-0.0768, -0.0434] | 0.25 | 9.73e-13 | 1.79e-72 |
| all | hnr_db | higher | -2.3847 [-2.6281, -2.1392] | 0.23 | 7.11e-15 | 1.29e-82 |
| all | stoi_trunc | higher | -0.0361 [-0.0414, -0.0309] | 0.30 | 1.42e-14 | 1.84e-46 |
| all | pesq_trunc | higher | 0.0104 [0.0046, 0.0162] | 0.65 | 0.00571 | 8.21e-13 |

## Paired: copysynth_tempo − distorted (per clip)

| severity | metric | better | mean diff [95% CI] | share copysynth_tempo better | p (Wilcoxon, speakers) | p (Wilcoxon, clips) |
|---|---|---|---|---|---|---|
| mild | cer | lower | -0.1522 [-0.1852, -0.1210] | 0.67 | 2.64e-12 | 9.81e-28 |
| mild | wer | lower | -0.2938 [-0.3452, -0.2451] | 0.65 | 2.4e-09 | 1.44e-27 |
| mild | stoi | higher | 0.0574 [0.0502, 0.0653] | 0.88 | 7.11e-15 | 4.49e-43 |
| mild | pesq | higher | 0.3994 [0.3676, 0.4328] | 1.00 | 7.11e-15 | 3.02e-52 |
| mild | mel_l1_clean | lower | -1.9057 [-2.1013, -1.7274] | 1.00 | 7.11e-15 | 2.99e-52 |
| mild | spk_sim_same | higher | 0.2928 [0.2776, 0.3082] | 1.00 | 7.11e-15 | 2.99e-52 |
| mild | spk_sim_other | higher | 0.1342 [0.1226, 0.1461] | 0.98 | 7.11e-15 | 4.59e-52 |
| mild | f0_corr | higher | 0.0136 [-0.0050, 0.0329] | 0.69 | 0.204 | 0.00119 |
| mild | f0_rmse_st | lower | -0.4545 [-0.5808, -0.3227] | 0.78 | 1.52e-06 | 2.99e-17 |
| mild | vuv_agree | higher | 0.0524 [0.0471, 0.0576] | 0.96 | 7.11e-15 | 8.92e-50 |
| mild | hnr_db | higher | 3.5003 [3.2040, 3.7858] | 0.98 | 7.11e-15 | 1.06e-51 |
| mild | stoi_trunc | higher | 0.0045 [-0.0047, 0.0141] | 0.48 | 0.532 | 0.943 |
| mild | pesq_trunc | higher | 0.6668 [0.6152, 0.7183] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | cer | lower | -0.3961 [-0.4528, -0.3444] | 0.92 | 7.11e-15 | 1.62e-45 |
| moderate | wer | lower | -0.6091 [-0.6932, -0.5363] | 0.86 | 7.11e-15 | 5.71e-43 |
| moderate | stoi | higher | 0.1697 [0.1613, 0.1783] | 1.00 | 7.11e-15 | 3.05e-52 |
| moderate | pesq | higher | 0.3800 [0.3471, 0.4156] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | mel_l1_clean | lower | -2.3152 [-2.5105, -2.1361] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | spk_sim_same | higher | 0.4790 [0.4626, 0.4962] | 1.00 | 7.11e-15 | 2.99e-52 |
| moderate | spk_sim_other | higher | 0.2429 [0.2254, 0.2602] | 0.99 | 7.11e-15 | 3.2e-52 |
| moderate | f0_corr | higher | 0.0773 [0.0544, 0.1015] | 0.76 | 2.3e-08 | 2.56e-14 |
| moderate | f0_rmse_st | lower | -1.5956 [-1.7282, -1.4615] | 0.91 | 7.11e-15 | 1.17e-43 |
| moderate | vuv_agree | higher | 0.0829 [0.0762, 0.0896] | 0.99 | 7.11e-15 | 1.02e-51 |
| moderate | hnr_db | higher | 5.4846 [5.1514, 5.8118] | 1.00 | 7.11e-15 | 3.05e-52 |
| moderate | stoi_trunc | higher | -0.0313 [-0.0392, -0.0234] | 0.26 | 6.35e-09 | 8.69e-14 |
| moderate | pesq_trunc | higher | 0.4174 [0.3826, 0.4545] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | cer | lower | -0.5541 [-0.6071, -0.5079] | 0.94 | 7.11e-15 | 1.44e-49 |
| severe | wer | lower | -0.7646 [-0.8418, -0.7024] | 0.92 | 7.11e-15 | 4.9e-47 |
| severe | stoi | higher | 0.2341 [0.2260, 0.2421] | 1.00 | 7.11e-15 | 3.02e-52 |
| severe | pesq | higher | 0.3731 [0.3440, 0.4049] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | mel_l1_clean | lower | -2.8960 [-3.0835, -2.7223] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | spk_sim_same | higher | 0.6361 [0.6242, 0.6488] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | spk_sim_other | higher | 0.3494 [0.3287, 0.3696] | 1.00 | 7.11e-15 | 3.14e-52 |
| severe | f0_corr | higher | 0.1884 [0.1479, 0.2354] | 0.83 | 2.13e-14 | 3.35e-29 |
| severe | f0_rmse_st | lower | -2.8167 [-3.0443, -2.6055] | 0.97 | 7.11e-15 | 3.54e-50 |
| severe | vuv_agree | higher | 0.1200 [0.1111, 0.1290] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | hnr_db | higher | 8.7690 [8.3205, 9.2082] | 1.00 | 7.11e-15 | 2.99e-52 |
| severe | stoi_trunc | higher | -0.0518 [-0.0624, -0.0408] | 0.24 | 2.53e-10 | 1.81e-22 |
| severe | pesq_trunc | higher | 0.3156 [0.2863, 0.3465] | 1.00 | 7.11e-15 | 2.99e-52 |
| all | cer | lower | -0.3675 [-0.4021, -0.3332] | 0.84 | 7.11e-15 | 6.8e-122 |
| all | wer | lower | -0.5558 [-0.6053, -0.5098] | 0.81 | 7.11e-15 | 1.53e-116 |
| all | stoi | higher | 0.1537 [0.1465, 0.1611] | 0.96 | 7.11e-15 | 3.11e-149 |
| all | pesq | higher | 0.3842 [0.3548, 0.4154] | 1.00 | 7.11e-15 | 8.28e-153 |
| all | mel_l1_clean | lower | -2.3723 [-2.5650, -2.1957] | 1.00 | 7.11e-15 | 8.25e-153 |
| all | spk_sim_same | higher | 0.4693 [0.4572, 0.4820] | 1.00 | 7.11e-15 | 8.25e-153 |
| all | spk_sim_other | higher | 0.2421 [0.2269, 0.2572] | 0.99 | 7.11e-15 | 1.51e-152 |
| all | f0_corr | higher | 0.0930 [0.0742, 0.1130] | 0.76 | 2.64e-12 | 7.04e-40 |
| all | f0_rmse_st | lower | -1.6210 [-1.7389, -1.5021] | 0.89 | 7.11e-15 | 6.47e-114 |
| all | vuv_agree | higher | 0.0851 [0.0796, 0.0906] | 0.98 | 7.11e-15 | 8.02e-151 |
| all | hnr_db | higher | 5.9180 [5.5917, 6.2336] | 0.99 | 7.11e-15 | 1.46e-152 |
| all | stoi_trunc | higher | -0.0262 [-0.0325, -0.0195] | 0.33 | 1.34e-08 | 7.64e-25 |
| all | pesq_trunc | higher | 0.4666 [0.4329, 0.5020] | 1.00 | 7.11e-15 | 8.25e-153 |
