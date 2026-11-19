# Trained weights of the final ensemble (submission 1054222)

The 18 checkpoints below are every model in the final ensemble. They are published as assets of the GitHub release
[`v1.0`](https://github.com/Arnavvvkumar/IEEE-BigData-2026-Cup---V2X-Track/releases/tag/v1.0) (622 MB in total).

Each file is a standard OpenPCDet checkpoint reduced to the inference keys (`model_state`, `epoch`, `it`, `version`);
the optimizer state was removed. `model_state` is tensor-for-tensor identical to the checkpoint used for the submission.
Load it with OpenPCDet's `model.load_params_from_file(path, logger)`.

Download and verify all files (SHA-256 against `manifest.json`):

```bash
bash weights/download_weights.sh            # into weights/ckpt/
```

| File | Config (`pipeline/canon/`) | Training set | Classes voted | Epoch | MB | SHA-256 |
|---|---|---|---|---|---|---|
| `vnx010_all_s1234_ep10.pth` | `vnx010_all.yaml` | `claude_canon_all_v1` | pedestrian, vehicle | 10 | 29.9 | `2f3a6392fb68fc7e?` |
| `vnx010_twin_s1_ep12.pth` | `vnx010_all.yaml` | `claude_canon_all_twin_v1` | pedestrian, vehicle | 12 | 29.9 | `51b7f8adcd1466b8?` |
| `vnx010_all_rs2_amp_ep12.pth` | `vnx010_all.yaml` | `claude_canon_all_v1` | pedestrian | 12 | 29.9 | `cbe168b7b426be25?` |
| `vnx010_pv_ft_ep5.pth` | `vnx010_all.yaml` | `claude_canon_all_twin_pv_v1` | vehicle | 5 | 29.9 | `edb14cdee69a0559?` |
| `vnx010_pv_ft2_ep5.pth` | `vnx010_all.yaml` | `claude_canon_all_twin_pv_v1` | vehicle | 5 | 29.9 | `b159f0f3b97f1e02?` |
| `vnx010_pv_ft4_ep4.pth` | `vnx010_all.yaml` | `claude_canon_all_twin_pv_v1` | vehicle | 4 | 29.9 | `f5522b6eb423a37d?` |
| `canon_all_s1234_ep12.pth` | `second_canon_all_v1.yaml` | `claude_canon_all_v1` | vehicle, truck | 12 | 21.4 | `25e09e764d615f6b?` |
| `second_twin_s1_ep12.pth` | `second_canon_all_v1.yaml` | `claude_canon_all_twin_v1` | vehicle, truck | 12 | 21.4 | `306b50a0660d4bab?` |
| `second_twin_s2_ep12.pth` | `second_canon_all_v1.yaml` | `claude_canon_all_twin_v1` | vehicle, truck | 12 | 21.4 | `961ef344270eadd8?` |
| `canon_relabel_s1234_ep12.pth` | `second_canon_v1.yaml` | `claude_canon_relabel_mix6000_v1` | vehicle, truck | 12 | 21.4 | `91896abf39b44cf1?` |
| `second_twin_st_s1_ep12.pth` | `second_canon_all_v1.yaml` | `claude_canon_all_twin_st_v1` | truck | 12 | 21.4 | `535dbf254ef82517?` |
| `second_twin_pv_s1_ep14.pth` | `second_canon_all_v1.yaml` | `claude_canon_all_twin_pv_v1` | truck | 14 | 21.4 | `8686f2f6fb411942?` |
| `pvrcnn_st_osc_ep6.pth` | `pvrcnn_all.yaml` | `claude_canon_all_twin_st_v1` | pedestrian, vehicle, truck | 6 | 52.5 | `85f3c843da796300?` |
| `pvrcnn_st_osc_s2_ep4.pth` | `pvrcnn_all.yaml` | `claude_canon_all_twin_st_v1` | pedestrian, vehicle | 4 | 52.5 | `ee6d7497e5534e91?` |
| `pvrcnn_ped_P1_ep5.pth` | `pvrcnn_ped.yaml` | `claude_canon_all_twin_st_v1` | pedestrian | 5 | 52.3 | `a859d43e9fd5e737?` |
| `pvrcnn_ped_P2_ep5.pth` | `pvrcnn_ped.yaml` | `claude_canon_all_twin_st_v1` | pedestrian | 5 | 52.3 | `2264a74980855207?` |
| `pvrcnn_ped_P3_ep5.pth` | `pvrcnn_ped.yaml` | `claude_canon_all_twin_st_v1` | pedestrian | 5 | 52.3 | `18307e61dcc73972?` |
| `pvrcnn_ped_P4_ep5.pth` | `pvrcnn_ped_aug.yaml` | `claude_canon_all_twin_st_v1` | pedestrian | 5 | 52.3 | `43a7b940fd2b03de?` |

All models were trained from scratch on synthetic data only (see the report and `pipeline/LEARNINGS.md`).
