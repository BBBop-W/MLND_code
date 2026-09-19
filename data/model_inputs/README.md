# Processed model inputs

These small source assets are sufficient to assemble the current model with
canonical site/OD CSVs. Solving does not read raw orders or experiment outputs.

- `rider_time.json`: fitted distance slope for first/last-mile rider travel and
  its observational scope. The working rider distribution is exponential.
- `H*_postfetch_deadlines.csv`: one row per canonical OD, in canonical order;
  observed counts, observed post-fetch means, prior and shrunk delivery deadline.
- `pre_dispatch_statistics.json`: historical payment-to-fetch summaries.
  The mean is informational and excluded from drone workload.
- `calibration.json`: rider acceptance delay, per-size source hashes, table
  hashes, calibration definitions and relative filenames.
- `manifest.json`: integrity hashes for the processed model-input files.

The first-mile travel mean is 5.733438163127652 minutes per straight-line
kilometer. The terminal rider-calling proxy is 4.519596062942166 minutes from
request creation to rider acceptance. Observed first-mile merchant-to-locker
travel and its distribution are not separately identified by these records.

Delivery deadlines use the observed `arrived_time - fetch_time` duration after
strict chronology filtering and an OD prior weight of 5. Payment-to-fetch time
is excluded. Missing OD means remain NaN while every deadline is positive.

Only source and calibration metadata were relocated. Numeric CSV tables are
unchanged and are checked against the canonical OD-coordinate and demand files
when loading an instance. Filenames inside the manifest are relative to this
directory; raw-source metadata paths are relative to the repository.
