# REX3 Forward First-Arrival Structural Path Analysis

A GPU-accelerated forward structural path analysis (SPA) framework for tracing commodity flows from specific origin countries to destination regions through the global inter-industry transaction network, with environmental impact attribution along each identified supply-chain route.

## Method

The framework combines three established input-output components in a configuration that, to our knowledge, has no single published precedent:

1. **Ghosh allocation matrix** ($B = \hat{x}^{-1}Z$) for forward propagation through observed inter-industry transactions (Ghosh 1958; Dietzenbacher 1997)
2. **First-arrival accounting** with destination-region absorption: when origin-linked flow first enters a destination-region node, it is captured and removed from further propagation
3. **Alpha coefficient** ($\alpha_i = Y_{\text{target},i} / x_i$) for final-consumption capture at each SPA depth

The Ghosh matrix is used strictly as a descriptive accounting device recording how each unit of a seller's output was allocated across intermediate buyers during the accounting period. No behavioural supply-push assumption is made, and the well-documented theoretical objections of Oosterhaven (1988) do not apply to this descriptive use.

### What It Answers

Given a specific origin country-sector (e.g., Indonesian vegetable oil processing), the method identifies:

- Which countries and sectors the output passes through before first arriving in the destination region (e.g., EEA)
- Whether arrival occurs as intermediate consumption by destination industries or final consumption by destination households and governments
- The tier depth (number of inter-industry transaction steps) of each route
- Whether routes are direct, domestic-indirect (origin-country value-chain processing), or third-country indirect
- Gate-to-gate and cradle-to-gate environmental impacts embedded in each route

### What It Does Not Answer

- How supply chains would respond to policy interventions (the method is attributional, not consequential)
- Certification-specific environmental performance within aggregate sectors
- Total domestic-plus-imported footprints (only imported first-arrival exposure)

## Data

The implementation uses **REX3**, a geographically resolved extension of the EXIOBASE 3 multi-regional input-output database (Stadler et al. 2018):

- **189 countries** (extended from EXIOBASE 3's 44 countries + 5 rest-of-world regions)
- **163 product sectors** with detailed environmental extensions (GHG, land use, water, materials)
- **Transaction matrix:** ~30,807 country-sector nodes (30,000+ before aggregation)
- **Aggregated matrix:** 25,917 x 25,917 after destination-region collapse (e.g., 31 EEA countries to 1)
- **Year:** 2022

REX3 was selected over GTAP (57-65 sectors), WIOD (43 countries, 56 sectors), and Eora (26 sectors harmonised) for its combination of high geographic and sectoral resolution with comprehensive environmental satellite accounts, including land-use-change and deforestation extensions.

## Pipeline

### Step 1: Parser and Denoiser

`Step 1 parser_denoise_HDF5.py`

Parses raw REX3 `.mat` files and caches the cleaned MRIO tables (Z, Y, x, S, satellite accounts) as HDF5. This avoids re-parsing 4+ GB source files on each run.

**Configuration:**

```python
BASE_PATH = r"REX3_Data"
YEARS_TO_PROCESS = [2022]
```

### Step 2: SPA Engine

`Step 2 RENEW SPA_Ghosh_Edge_Depth_Tracking.py`

Core GPU-accelerated SPA engine. Computes Leontief inverse L, Ghosh allocation matrix B, alpha coefficients, preprocesses B for first-arrival boundary conditions, and executes batched SPA with full edge-depth tracking.

**Configuration:**

```python
years = [2022]
max_depth = 100
threshold = 1e-16
min_output_threshold = 1e-3   # 0.001 M EUR

origin_countries = ["Indonesia"]
target_groups = {
    "EEA": ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus",
             "Czech Republic", "Denmark", "Estonia", "Finland", "France",
             "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy",
             "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
             "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
             "Spain", "Sweden", "Norway", "Switzerland", "Liechtenstein"]
}
sectors = ["Processing vegetable oils and fats"]
```

**Key operations:**

- Leontief inverse: $L = (I - A)^{-1}$ on aggregated 25,917 x 25,917 matrix
- Ghosh allocation: $B = \hat{x}^{-1}Z$ with target-to-target zeroing
- Alpha vector: $\alpha_i = Y_{\text{target},i} / x_i$ for final-consumption capture
- Batched SPA iteration with per-step IC/FC capture, edge recording, and convergence checking
- C2G intensity: $\text{C2G}_{k} = \sum_i S_{ki} \times L_{i,\text{origin}}$
- G2G intensity: $\text{G2G}_{k} = S_{k,\text{origin}}$

**Output:** Parquet files with full edge-depth-route records, Excel summaries, checkpoint CSVs.

### Step 3: Visualizer

`Step 3 Data Visualizer V2.py`

Reads Step 2 output and generates Sankey diagrams and summary tables for route analysis.

## Hardware Requirements

### GPU (required)

The SPA engine uses CuPy for all matrix operations and requires an NVIDIA GPU with CUDA support.

| Operation                           | Peak VRAM                                  |
| ----------------------------------- | ------------------------------------------ |
| Matrix inversion (L, G on 25,917²) | ~16.1 GB                                   |
| SPA batched iteration               | ~5.6 GB                                    |
| **Minimum GPU VRAM**          | **24 GB (e.g., L4, RTX 4090, A100)** |

### CPU RAM

| Operation                         | Peak RAM                    |
| --------------------------------- | --------------------------- |
| Loading full DataFrames from HDF5 | ~45 GB                      |
| **Minimum system RAM**      | **64 GB recommended** |

### Precision

FP64 (double precision) is mandatory throughout the pipeline. FP32 produces unreliable results due to error cascade through 100-depth SPA iterations, condition-number sensitivity in 25,917² matrix inversion, small-denominator amplification in the alpha vector, and C2G/G2G divergence magnification (ratios up to 4,500x for land-use-change emissions).

## Completed Analyses

All analyses use 2022 REX3 data.

| Origin    | Destination | Sector                             | Status   |
| --------- | ----------- | ---------------------------------- | -------- |
| Indonesia | EEA         | Processing vegetable oils and fats | Complete |
| Indonesia | EEA         | EUDR commodity basket              | Complete |
| Indonesia | EEA         | Paper                              | Complete |
| Indonesia | EEA         | Pulp                               | Complete |
| Indonesia | EEA         | Cultivation of oil seeds           | Complete |
| Indonesia | China       | Processing vegetable oils and fats | Complete |
| Brazil    | EEA         | EUDR commodity basket              | Complete |
| Brazil    | EEA         | Processing of meat cattle          | Complete |

**Important labeling note:** For Brazil-origin scenarios, the "Processing vegetable oils and fats" sector is approximately 90% soybean oil, not palm oil. Results must be labeled as "Processing vegetable oils and fats (predominantly soy)" to avoid materially misleading policy interpretations (Decision D18).

## Repository Structure

```
project_registry/
  eudr_products/           # EUDR methodology, sources, results, audit documents
    audit/                 # 5 external audit sessions (3 model families)
    results/               # Completed analysis outputs (parquet, Excel, Sankey)
    sources/literature/    # Literature review files and master bibliography
    methodology_paper_sections.md  # Publication-draft methodology
  shared/
    code/                  # Pipeline scripts (Step 1-3)
    communications/        # Development chat logs
  tuna/                    # Separate tuna project
REX3_Data/                 # Raw MRIO matrices (27 GB, untracked)
docs/                      # Project state, decisions, session logs
PM_Log/                    # Project management artifacts
```

## Methodology Validation

The methodology has been subjected to five independent external audits across three model families:

| Audit                                    | Focus                                        | Verdict                                                           |
| ---------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------- |
| Comprehensive methodology + code         | Core methodology V1-V7, code alignment C1-C7 | Components established; hybrid configuration novel-but-defensible |
| Empirical data validation                | Data R1-R5, convergence patterns             | Core matrices match methodology                                   |
| Hostile adversarial review (GPT-5.4)     | Literature precedent, alpha uniqueness       | No single published precedent found; alpha is unique choice       |
| Hostile adversarial review (Antigravity) | Cross-audit consistency                      | Publishable with revisions                                        |
| Phase 2 literature audit                 | Citation verification, bibliography          | 54 citations, 53 verified                                         |

**Consolidated verdict:** Publishable with revisions. The method uses established components (Ghosh B, Leontief L, SPA power series, C2G intensities) in a novel forward first-arrival configuration.

## Key References

Defourny, J. and Thorbecke, E. (1984). Structural path analysis and multiplier decomposition within a social accounting matrix framework. *The Economic Journal*, 94(373), 111-136. DOI: 10.2307/2232220.

Dietzenbacher, E. (1997). In vindication of the Ghosh model: a reinterpretation as a price model. *Journal of Regional Science*, 37(4), 629-651. DOI: 10.1111/0022-4146.00073.

Ghosh, A. (1958). Input-output approach in an allocation system. *Economica*, 25(97), 58-64. DOI: 10.2307/2550694.

Miller, R.E. and Blair, P.D. (2009). *Input-output analysis: foundations and extensions*. 2nd edn. Cambridge: Cambridge University Press. DOI: 10.1017/CBO9780511626982.

Oosterhaven, J. (1988). On the plausibility of the supply-driven input-output model. *Journal of Regional Science*, 28(2), 203-217. DOI: 10.1111/j.1467-9787.1988.tb01208.x.

Peters, G.P. (2008). From production-based to consumption-based national emission inventories. *Ecological Economics*, 65(1), 13-23. DOI: 10.1016/j.ecolecon.2007.10.014.

Stadler, K. et al. (2018). EXIOBASE 3: developing a time series of detailed environmentally extended multi-regional input-output tables. *Journal of Industrial Ecology*, 22(3), 502-515. DOI: 10.1111/jiec.12715.

Wiedmann, T. (2009). A review of recent multi-region input-output models used for consumption-based emission and resource accounting. *Ecological Economics*, 69(2), 211-222. DOI: 10.1016/j.ecolecon.2009.08.026.

## License

This project is not yet licensed for public use. Contact the author for collaboration inquiries.

## Author

Izzu Prawiranegara
