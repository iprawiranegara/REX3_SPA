# oil-split-spa

`oil-split-spa` is **REX3 structural path analysis for disaggregated vegetable oil**.
Its Python package is `oil_split_spa`.

The application runs after `rex3-vegetable-oil-builder` has created and
independently validated a local vegetable-oil-disaggregated REX3 database and
its matching vegetable-oil rebuild receipt.

## Required input

The application takes:

- the accepted vegetable-oil-disaggregated REX3 HDF5 database;
- its matching vegetable-oil rebuild receipt;
- the builder release version; and
- an analysis configuration.

Before a full matrix data read, preflight reads HDF5 root metadata and
compares the receipt and HDF5 map bindings. It then checks the schema,
dimensions, order-sensitive axes, year, fingerprints, and required validation
outcomes. A missing field, changed fingerprint, or failed required check stops
the run.

The receipt is the boundary between the builder and this application. The
application does not resolve private mappings, search a user's computer, or
load a copied cache.

## Analysis command and configuration

```text
oil-split-spa run --database vegetable-oil.h5 --receipt vegetable-oil-rebuild-receipt.json --release-version 0.1.0 --config ANALYSIS_CONFIG.json
```

The analysis configuration accepts only these fields:

- `year`, which must be 2022;
- `final_demand_labels`;
- `impact_indicators`;
- `cutoff`;
- `max_depth`; and
- `output_directory`.

The output directory is selected by the user and should be outside the source
tree. The command also accepts `--output-directory` to override the configured
directory and `--allow-test-fixture` for the explicit synthetic profile used
by tests.

Each run writes exactly these three deterministic names:

- `summary.json`;
- `path-rows.json`; and
- `application-run-receipt.json`.

The application receipt records the builder receipt and database fingerprints,
configuration fingerprint, and output fingerprints.

`path-rows.json` contains depth and impact aggregates for each selected impact
indicator. It does not contain country-sector sequences or trade routes.

## Builder boundary

Run the builder commands in this order before analysis:

```text
rex3-vegetable-oil build-base --config RAW_INPUT_CONFIG.json --output BASE.h5 --receipt BASE_RECEIPT.json
rex3-vegetable-oil import-allocation --trade-source TRADE_SOURCE.csv --production-source PRODUCTION_SOURCE.csv --output-dir ALLOCATION_DIR
rex3-vegetable-oil prepare-weights --trade ALLOCATION_DIR/trade-allocation.csv --production ALLOCATION_DIR/production-allocation.csv --countries COUNTRY_1,COUNTRY_2 --output WEIGHTS.json
rex3-vegetable-oil build --config BUILD_CONFIG.json
rex3-vegetable-oil validate --database vegetable-oil.h5 --receipt vegetable-oil-rebuild-receipt.json
```

The final rebuild receipt retains exactly eleven top-level fields:
`receipt_schema`, `release_version`, `command_name`,
`configuration_fingerprint`, `source_records`, `output`, `checks`,
`optional_checks`, `backend_identity`, `software_identity`, and `fallbacks`.
The separate base-build receipt from `build-base` is not accepted in place of
the final vegetable-oil rebuild receipt.

Production builder output binds the named release maps
`country-concordance.json`, `item-map.json`, and
`vegetable-oil-scope.json`. The builder validates and hashes them, stores the
binding in the final receipt and HDF5 root metadata, and includes it in the
configuration fingerprint. The country concordance has 187 mapped and 2
declared unavailable REX3 country identifiers. `import-allocation` and
`prepare-weights` use zero shares only for those two rows. An observed country
or bilateral pair must have a positive source total; otherwise the importer
stops without a fallback share.

## Optional CuPy backend

`oil-split-spa` uses NumPy for its analysis and does not install or select
CuPy. It accepts a builder receipt that records either the `numpy` or
`cupy-managed` build backend.

## Sources and package boundary

REX3 2022 and the needed FAOSTAT trade and crop-production inputs are
user-acquired and never bundled. Review publisher use and reuse terms at
acquisition time. Access does not grant redistribution rights.

USDA Production, Supply and Distribution and UN Comtrade are external
reference-only sources. Their links and terms remain reference information.
They are not builder inputs and do not appear in the analysis configuration,
receipt, or validation results.

No raw data, HDF5/database files, tensors, inverse matrices, caches, generated
scientific results, credentials, or local paths are package files. Generated
analysis output is written only to the user-selected output directory.

## Checks and package build

Run the application tests with bytecode and pytest-cache writes disabled:

```bash
env PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
```

Build package archives into a directory outside the source tree:

```bash
python -m build --sdist --wheel --outdir "$BUILD_DIR"
```

## License

Copyright (c) 2026 Izzu Prawiranegara. This package's source code is available
under the MIT License. The license applies to this package's code and
documentation, not to user-acquired REX3, FAOSTAT, or other source inputs.
