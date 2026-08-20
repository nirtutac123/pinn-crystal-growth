# CFD Surrogate GPR Results

These outputs were generated from the corrected Czochralski CFD raw CSV data using:

```bash
python experiments/train_cfd_surrogate.py --dataset temperature --data_root /tmp/cz_corrected --holdout 1780.0 --max_rows_per_case 60
python experiments/train_cfd_surrogate.py --dataset crucible --data_root /tmp/cz_corrected --holdout -4.0 --max_rows_per_case 60
python experiments/train_cfd_surrogate.py --dataset crystal --data_root /tmp/cz_corrected --holdout 7.0 --max_rows_per_case 60
```

The absolute `/tmp/cz_corrected` path is only the local extraction location used during generation. The same commands work with any extracted corrected data zip by replacing `--data_root`.

## Quick Comparison

| Dataset | Holdout | Temperature relative RMSE | Hardest observed component |
| --- | --- | ---: | --- |
| Temperature sweep | `1780 K` | `0.0561` | `u_z` relative RMSE `0.2785` |
| Crucible rotation sweep | `-4 rpm` | `0.0634` | `u_z` relative RMSE `0.1346` |
| Crystal rotation sweep | `7 rpm` | `0.0958` | `u_z` relative RMSE `0.4146` |

The crystal-rotation holdout is the most difficult of the three compact GPR checks. This is useful for the paper because it shows where uncertainty and flow-sensitive behaviour should be discussed explicitly.
