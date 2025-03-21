# REX3 SPA Analysis Framework

## Multi-Regional Input-Output Structural Path Analysis with GPU Acceleration

This repository contains a GPU-accelerated implementation of Structural Path Analysis (SPA) for analyzing environmental and economic impacts along global supply chains using Multi-Regional Input-Output (MRIO) modeling. The framework is specifically designed to quantify how environmental impacts are embedded in international trade flows between regions and sectors.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Multi-Regional Input-Output Foundation](#multi-regional-input-output-foundation)
  - [Data Preprocessing](#data-preprocessing)
  - [Structural Path Analysis Implementation](#structural-path-analysis-implementation)
  - [Environmental Impact Calculation](#environmental-impact-calculation)
  - [Cradle-to-Gate Calculations](#cradle-to-gate-calculations)
- [Configuration](#configuration)
- [Output Description](#output-description)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

## Features

- **GPU-accelerated Structural Path Analysis** for efficient processing of large MRIO systems
- **Multi-country target aggregation** for analyzing country groups (e.g., EU27) as single entities
- **Comprehensive environmental impact assessment** covering climate change, material footprint, water stress, biodiversity loss, and more
- **Resilient execution framework** with checkpoint system for resumable analysis
- **Memory-efficient processing** for handling large datasets
- **Multi-stream CUDA execution** for parallel computation
- **Detailed trade flow decomposition** into direct and indirect components

## Requirements

- Python 3.7+
- NVIDIA GPU with CUDA support
- Required Python packages:
  - numpy
  - pandas
  - cupy
  - openpyxl
  - psutil
  - tqdm

## Installation

1. Clone this repository:
```bash
git clone https://github.com/username/rex3-spa-analysis.git
cd rex3-spa-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the paths in the main script to point to your data directory:
```python
pickle_path_template = r"REX3_Data/cached_rex3_data/rex3_{year}.pkl"
output_directory = r"REX3_Results"
```

## Usage

1. Configure the parameters in the user parameters section of the script:
```python
# User parameters
years = [2018, 2019, 2020, 2021, 2022]
num_streams = 4  # Adjust based on GPU capabilities
max_depth = 1000
threshold = 1e-16
# Define target groups and sectors
```

2. Run the main script:
```bash
python rex3_spa_analysis.py
```

3. Results will be saved in the specified output directory as both CSV checkpoints and an Excel file.

## Methodology

### Multi-Regional Input-Output Foundation

The analysis is built on MRIO modeling, which represents the global economy as an interconnected network of industries and regions. The core matrices include:

- **Z**: Intermediate consumption matrix showing inter-industry transactions
- **Y**: Final demand matrix showing consumption by end-users
- **x**: Total output vector
- **A**: Technical coefficients matrix (A = Z·x̂⁻¹)
- **L**: Leontief inverse matrix (L = (I-A)⁻¹)
- **F**: Environmental extensions matrix (direct environmental impacts)
- **S**: Stressor intensity matrix (S = F·x̂⁻¹)

A custom `MRIOTable` dataclass encapsulates these matrices:

```python
@dataclass
class MRIOTable:
    """Stores MRIO matrices and environmental extensions."""
    year: int
    Z: Optional[pd.DataFrame] = None  # Transaction matrix
    Y: Optional[pd.DataFrame] = None  # Final demand
    x: Optional[pd.DataFrame] = None  # Total output
    A: Optional[pd.DataFrame] = None  # Technical coefficients
    F: Optional[pd.DataFrame] = None  # Environmental stressors
    S: Optional[pd.DataFrame] = None  # Direct stressor coefficients
    L: Optional[pd.DataFrame] = None  # Leontief inverse
    Q: Optional[pd.DataFrame] = None  # Raw environmental stressors
    meta: Dict[str, Any] = field(default_factory=dict)
```

### Data Preprocessing

#### Methodological Rationale

1. **Prevention of Double-Counting**: Without preprocessing, flows would circulate within target regions before being counted, leading to inflated estimates.

2. **First Arrival Principle**: The analysis aims to capture the "first arrival" of goods/services into the target region, not subsequent internal trades.

3. **Clean Separation of Flows**: This ensures a clear distinction between direct imports and domestically produced goods within the target region.

#### Matrix Preprocessing for A Matrix

To properly account for trade flows, the code preprocesses the A matrix to zero out target-to-target connections. This methodological step is crucial for correctly identifying flows from non-target to target regions.

```python
def preprocess_A_matrix(A_values, target_mask):
    # Create copy to avoid modifying original
    A_preprocessed = np.copy(A_numpy)
    
    # Create masks for target-to-target connections
    row_mask = np.repeat(target_mask[:, np.newaxis], A_preprocessed.shape[1], axis=1)
    col_mask = np.repeat(target_mask[np.newaxis, :], A_preprocessed.shape[0], axis=0)
    target_to_target = row_mask & col_mask
    
    # Zero out target-to-target connections
    A_preprocessed[target_to_target] = 0.0
```

The implementation creates two boolean masks:
- `row_mask`: A 2D mask where rows corresponding to target regions are True
- `col_mask`: A 2D mask where columns corresponding to target regions are True

These are combined with a logical AND operation to identify all target-to-target cells in the A matrix, which are then set to zero.

#### Country Aggregation Framework

The code implements a sophisticated framework to handle multi-country targets like the EU27 by aggregating them into a single entity while preserving sector detail.

```python
def aggregate_multiregion_target(rex3, target_countries, year, target_groups=target_groups, force_recompute=False):
    # Generate target key for this group
    target_key = get_target_key(target_countries, target_groups)
    
    # Define paths for temporary files
    Z_path = temp_path / f"Z_{target_key}_{year}.pkl"
    Y_path = temp_path / f"Y_{target_key}_{year}.pkl"
    # ... (paths for other matrices)
    
    # Check for cached matrices
    all_cached = all(p.exists() for p in [Z_path, Y_path, x_path, A_path, L_path])
    if all_cached and not force_recompute:
        # Load cached matrices
        # ...
    else:
        # Perform aggregation
```

##### Matrix Aggregation Process

The aggregation involves several carefully orchestrated steps:

###### 1. Z Matrix Aggregation (Intermediate Transactions)

```python
def aggregate_matrix_rows(matrix, target_countries):
    target_key = get_target_key(target_countries)
    
    # Create new index where target countries are replaced by aggregated entity
    new_index = []
    for idx in matrix.index:
        country, sector = idx[0], idx[1]
        if country in target_countries:
            new_index.append((f"AGG_{target_key}", sector))
        else:
            new_index.append((country, sector))
    
    # Create copy with new index and sum duplicate entries
    result = matrix.copy()
    result.index = pd.MultiIndex.from_tuples(new_index, names=matrix.index.names)
    result = result.groupby(level=matrix.index.names).sum()
    
    return result
```

This function transforms the rows of a matrix, replacing individual target countries with a single aggregated entity while preserving sector detail. The columns are aggregated similarly in a separate function.

###### 2. Y Matrix Special Handling

```python
def aggregate_Y_matrix_cols(matrix, target_countries):
    # Similar to normal column aggregation but handles Y's different structure
    target_key = get_target_key(target_countries)
    
    # Create new columns with target countries replaced
    new_cols = []
    for col in matrix.columns:
        col_values = list(col)
        country = col_values[0]  # First level (Header1) is typically country
        if country in target_countries:
            col_values[0] = f"AGG_{target_key}"
        new_cols.append(tuple(col_values))
    
    # Create a copy with the new columns
    result = matrix.copy()
    result.columns = pd.MultiIndex.from_tuples(new_cols, names=matrix.columns.names)
    
    # Sum values for columns with the same structure
    result = result.groupby(level=list(range(len(matrix.columns.names))), axis=1).sum()
    
    return result
```

The Y matrix has a different column structure that requires special handling during aggregation.

###### 3. Derived Matrix Recalculation

After aggregating the base matrices (Z, Y, F), the derived matrices are recalculated:

- **x vector**: Sum of aggregated Z rows and Y rows
- **A matrix**: Normalized by dividing aggregated Z by aggregated x
- **L matrix**: Leontief inverse of the aggregated A
- **S matrix**: Normalized by dividing aggregated F by aggregated x

##### Computational Efficiency and Memory Management

The aggregation framework implements several optimizations:

1. **Persistent Caching**: Saves aggregated matrices to disk for reuse in future runs
2. **Memory-Efficient Processing**: Processes and saves each matrix sequentially
3. **Smart Validation**: Checks cache age and option for forced recomputation

This approach enables efficient analysis of multi-country regions, treating them as single entities for both direct exports and SPA calculations.

### Structural Path Analysis Implementation

#### Mathematical Foundation

In standard input-output analysis, the total outputs required to meet a final demand is given by:

x = (I-A)⁻¹ · y = L · y

Where:
- x is the total output vector
- A is the technical coefficients matrix
- L is the Leontief inverse matrix (I-A)⁻¹
- y is the final demand vector

SPA decomposes this total effect (L) into specific pathways. The Leontief inverse can be expanded as:

L = I + A + A² + A³ + ... + Aⁿ + ...

Where:
- I represents direct production
- A represents first-tier suppliers
- A² represents second-tier suppliers
- And so on

#### Alpha Array Construction for SPA Final

The alpha array is a crucial component that determines what fraction of each sector's output goes to final demand in the target region, used specifically in the SPA Final calculation.

```python
def build_alpha_final(rex3: MRIOTable, target_countries, threshold: float):
    # Calculate global final demand
    Y_global_sum = rex3.Y.sum(axis=1)
    
    # Calculate total output needed to satisfy this demand
    x_baseline = rex3.L.dot(Y_global_sum)
    
    # Identify and sum final demand from target countries
    if is_multiregion_target(target_countries):
        target_key = get_target_key(target_countries)
        agg_target = f"AGG_{target_key}"
        mask_target = rex3.Y.columns.get_level_values('Header1').isin([agg_target])
    else:
        mask_target = rex3.Y.columns.get_level_values('Header1').isin(target_countries)
    
    Y_target = rex3.Y.loc[:, mask_target].sum(axis=1)
    
    # Calculate alpha as the ratio of target final demand to total output
    denom = x_baseline.values
    denom_safe = np.where(denom > threshold, denom, np.inf)
    alpha = Y_target.values / denom_safe
    alpha[~np.isfinite(alpha)] = 0.0
    
    return alpha
```

##### Mathematical Interpretation

The alpha value for each sector represents:
α = Y_target / x_baseline

Where:
- Y_target is the final demand from target countries for a sector's output
- x_baseline is the total output of that sector to satisfy all global demand

This ratio indicates what fraction of each unit of output eventually serves final demand in the target region.

##### Numerical Safeguards

The code includes safeguards to handle numerical issues:
- Values below the threshold are treated as infinite to avoid division by very small numbers
- Non-finite values (NaN, infinity) are replaced with zeros
- Explicit thresholding ensures numerical stability throughout calculations

#### Core SPA Functions

The code implements two core SPA functions that operate on the GPU for computational efficiency:

##### 1. Intermediate Consumption Paths

```python
def spa_intermediate_cuda(A_gpu, origin_idx, target_mask, max_depth, threshold, stream=None):
    # Initialize flow vector with 1.0 at origin position
    flow_vec = cp.zeros(N, dtype=cp.float64)
    flow_vec[origin_idx] = 1.0
    target_mask_gpu = cp.asarray(target_mask)
    total_flow = 0.0
    
    with stream or cp.cuda.Stream.null:
        for d in range(1, max_depth + 1):
            # Push flow through one more tier of supply chain
            flow_vec = cp.dot(A_gpu, flow_vec)
            
            # Numerical stability: threshold small values to zero
            flow_vec[cp.abs(flow_vec) < threshold] = 0.0
            
            # Calculate flow entering target regions
            flow_in = cp.sum(flow_vec[target_mask_gpu]).item()
            total_flow += flow_in
            
            # Zero out flow in target regions to prevent double-counting
            flow_vec[target_mask_gpu] = 0.0
            
            # Check for convergence
            if cp.all(flow_vec == 0.0):
                break
    
    return total_flow
```

This function traces how much of the output from a specific origin sector eventually reaches the target region through intermediate consumption. The algorithm:

1. Starts with a unit of output from the origin
2. Iteratively propagates this flow through the supply chain using the A matrix
3. At each step, captures the flow entering the target region
4. Zeros out target entries to prevent counting internal circulation
5. Continues until reaching maximum depth or convergence

##### 2. Final Demand Paths

```python
def spa_final_cuda(A_gpu, origin_idx, alpha_final, target_mask, max_depth, threshold, stream=None):
    # Similar initialization as intermediate function
    flow_vec = cp.zeros(N, dtype=cp.float64)
    flow_vec[origin_idx] = 1.0
    alpha_final_gpu = cp.asarray(alpha_final)
    target_mask_gpu = cp.asarray(target_mask)
    total_flow_fd = 0.0
    
    with stream or cp.cuda.Stream.null:
        for d in range(1, max_depth + 1):
            # Identify non-zero flows
            mask = cp.abs(flow_vec) >= threshold
            if cp.any(mask):
                # Calculate portion going to final demand in target
                fd = flow_vec[mask] * alpha_final_gpu[mask]
                flow_in_fd = cp.sum(fd).item()
                total_flow_fd += flow_in_fd
                
                # Remove final demand portion from flow
                flow_vec[mask] -= fd
            
            # Zero out target entries and continue propagation
            flow_vec[target_mask_gpu] = 0.0
            flow_vec = cp.dot(A_gpu, flow_vec)
            flow_vec[cp.abs(flow_vec) < threshold] = 0.0
            
            # Check for convergence
            if cp.all(flow_vec == 0.0):
                break
    
    return total_flow_fd
```

This function calculates how much of the origin's output reaches the target through final demand. The key difference is that at each step, a portion of the flow (determined by the alpha array) is allocated to final demand before the remainder continues through the supply chain.

##### Combined SPA Function

```python
def spa_two_measures(A_gpu, origin_idx, target_mask, alpha_final, max_depth, threshold, stream=None):
    try:
        gc.collect()
        if origin_idx is None:
            return (0.0, 0.0)
        frac_int = spa_intermediate_cuda(A_gpu, origin_idx, target_mask, max_depth, threshold, stream)
        frac_fd = spa_final_cuda(A_gpu, origin_idx, alpha_final, target_mask, max_depth, threshold, stream)
        
        return (frac_int, frac_fd)
    finally:
        gc.collect()
```

This wrapper function calls both SPA calculations for a given origin sector and returns both results, enabling the calculation of both intermediate and final demand-related flows in a single operation.

### Environmental Impact Calculation

Environmental impacts are calculated by combining trade flows with impact coefficients:

```python
def precompute_F_vectors(rex3: MRIOTable, impact_keys):
    """
    Precompute F vectors for all environmental indicators.
    """
    F_vectors = {}
    for impact_group, impact_data in impact_keys.items():
        indicators = impact_data["indicators"]
        conversion = impact_data.get("conversion", None)
        F_vectors[impact_group] = pd.Series(0.0, index=rex3.F.columns)
        for indicator in indicators:
            try:
                conversion_factor = conversion.get(indicator, 1.0) if conversion else 1.0
                F_vector = find_F_vector_for_indicator(rex3.F, indicator, conversion_factor)
                F_vectors[impact_group] += F_vector
            except Exception as e:
                logger.warning(f"Error processing indicator {indicator}: {e}")
    return F_vectors
```

For each indicator, the code:
1. Extracts the relevant row from the F matrix
2. Applies any necessary conversion factors
3. Combines multiple indicators if needed for an impact category
4. Creates a vector ready for multiplication with trade flow fractions

### Cradle-to-Gate Calculations

The framework calculates cradle-to-gate environmental impacts for each origin-sector pair and allocates them across different trade flow types:

```python
# Calculate environmental impacts
for impact_group, F_vector_total in precomputed_F_vectors.items():
    try:
        L_col = rex3_L.loc[:, (origin, sector)]
        cradle2gate_total = np.sum(F_vector_total.values * L_col.values)
    except KeyError:
        cradle2gate_total = 0.0

    if tot_out > 0:
        frac_dirFC = direct_FC / tot_out
        frac_dirIC = direct_IC / tot_out
        frac_spaIC = spa_IC / tot_out
        frac_spaFC = spa_FC / tot_out
    else:
        frac_dirFC = frac_dirIC = frac_spaIC = frac_spaFC = 0.0

    dirFC_c2g = cradle2gate_total * frac_dirFC
    dirIC_c2g = cradle2gate_total * frac_dirIC
    spaIC_c2g = cradle2gate_total * frac_spaIC
    spaFC_c2g = cradle2gate_total * frac_spaFC

    results[f"C2G TotalOutput - ({impact_group})"] = cradle2gate_total
    results[f"C2G Direct Total - ({impact_group})"] = dirIC_c2g + dirFC_c2g
    results[f"C2G Indirect Total - ({impact_group})"] = spaIC_c2g + spaFC_c2g
    results[f"C2G ExportTotal - ({impact_group})"] = (dirFC_c2g + dirIC_c2g + spaIC_c2g + spaFC_c2g)
```

The methodological steps for cradle-to-gate calculations are:

1. **Total Impact Calculation**: Multiply the impact intensity vector (F) by the Leontief column to get total cradle-to-gate impacts
2. **Flow Fraction Calculation**: Calculate what fraction of total output goes through each flow path (direct final, direct intermediate, SPA final, SPA intermediate)
3. **Impact Allocation**: Allocate total impact proportionally across flow paths
4. **Results Storage**: Store both total impacts and allocated impacts for reporting

This approach ensures that environmental burdens are correctly assigned to the trade flows that drive them, allowing for accurate attribution of impacts to consumption in target regions.

## Configuration

The main configuration parameters are defined at the top of the script:

```python
# Years to analyze
years = [2018, 2021, 2022, 2019, 2020]

# Input/output paths
pickle_path_template = r"REX3_Data/cached_rex3_data/rex3_{year}.pkl"
output_directory = r"REX3_Results"

# SPA parameters
max_depth = 1000
threshold = 1e-16

# Target groups definition
target_groups = {
    "China": ["China"],
    "EU27": ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", 
             "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy",
             "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", 
             "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
}

# Selected sectors
sectors = [
    "Processing vegetable oils and fats",
    "Cultivation of oil seeds",
    # Add more sectors as needed
]

# Impact keys mapping
impact_keys = {
    "Climate change impacts (kt CO2-eq)": {
        "indicators": [("Climate change impacts [kg CO2-eq]", "")],
        "conversion": {("Climate change impacts [kg CO2-eq]", ""): 1e-6}
    },
    "Total material footprint [kt]": {
        "indicators": [("Total material footprint [kt]", "")],
        "conversion": {("Total material footprint [kt]", ""): 1.0}
    },
    # Additional impact categories...
}
```

## Output Description

The analysis produces two main outputs:

1. **CSV Checkpoint File**: Contains detailed results for each origin-sector-target-year combination:
   - Direct and indirect export values
   - SPA intermediate and final consumption exports
   - Environmental impact indicators for each flow type

2. **Excel Summary**: Consolidated results with the same structure as the CSV checkpoint.

Each result row includes:
- Year, Target Group, Exporting Region, Exporting Sector
- Direct Intermediate/Final Export values
- SPA Intermediate/Final Export values
- Environmental impact indicators for multiple categories

## Performance Optimization

The framework implements several performance optimizations:

1. **GPU Acceleration with Multi-Stream Execution**:
```python
def create_stream_pool(n_streams):
    """Create pool of CUDA streams for concurrent execution on GPU."""
    return [cp.cuda.Stream() for _ in range(n_streams)]
```

2. **Memory Management**:
```python
# Free GPU memory after batch processing
cp.get_default_memory_pool().free_all_blocks()

# Monitor memory usage
log_cpu_memory("After processing batch")
log_gpu_memory("After loading matrices")
```

3. **Asynchronous Result Writing**:
```python
def checkpoint_writer_thread(checkpoint_path):
    """Background thread for writing batched results."""
    while not checkpoint_shutdown.is_set() or not result_queue.empty():
        results_batch = result_queue.get(timeout=1.0)
        batch_df = pd.DataFrame(results_batch)
        batch_df.to_csv(f, index=False, header=not file_exists)
```

4. **Matrix Caching**: Saves intermediate aggregated matrices to disk for reuse in subsequent runs.

5. **Dynamic Task Filtering**:
```python
def task_already_done(checkpoint_df, year, group_name, origin, sector):
    """Check if a specific task is already completed in the checkpoint."""
    if checkpoint_df.empty or 'Year' not in checkpoint_df.columns:
        return False
    mask = (
        (checkpoint_df['Year'] == year) &
        (checkpoint_df['Target Group'] == group_name_str) &
        (checkpoint_df['Exporting Region'] == origin_str) &
        (checkpoint_df['Exporting Sector'] == sector_str)
    )
    return mask.any()
```

## Contributing

Contributions to improve the framework are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request


---

*This REX3 SPA Analysis Framework was developed to support research on environmental impacts embedded in international trade.*
