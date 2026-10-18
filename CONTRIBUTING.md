# Contributing

Changes should be made on a short-lived branch and submitted through a pull
request. Keep each pull request focused on one experiment, model, dataset
change, or documentation improvement.

Before opening a pull request, run:

```bash
python -m compileall -q core equations examples experiments models scripts tools utils visualization main.py
python -m unittest discover -s experiments/tests -v
```

For changes to numerical results, include the command, random seed, dataset
version or DOI, evaluation split, and generated metric files. Do not commit
private CFD data, credentials, large checkpoints, or outputs whose
redistribution rights have not been confirmed.
