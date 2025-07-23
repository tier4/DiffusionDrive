# Bench2Drive Training NaN Investigation

## Executive Summary

Analysis of 10,000 cached samples from both NavSim and Bench2Drive datasets reveals significant differences that may contribute to NaN losses during training.

## Cache Data Comparison Results

### 1. Status Feature Differences
**NavSim**: min=-3.31, max=15.49, mean=0.696, std=1.94  
**Bench2Drive**: min=-26.97, max=14.32, mean=0.458, std=2.05

- Bench2Drive has an extreme minimum value of -26.97
- This is approximately 8x more negative than NavSim's minimum

### 2. Heading Distribution Differences  
**NavSim**: min=-1.65, max=1.44 radians, mean=0.050, std=0.233  
**Bench2Drive**: min=-0.110, max=0.110 radians, mean=-0.0006, std=0.017

- Bench2Drive heading range is 15x smaller than NavSim
- Standard deviation is 14x smaller
- 75th percentile is 0.00004 radians (near zero)
- Range appears symmetric at exactly ±0.110

### 3. Trajectory X Coordinate Differences
**NavSim**: min=-0.096, max=61.36 meters, mean=9.87  
**Bench2Drive**: min=-39.28, max=40.67 meters, mean=-0.082

- NavSim shows forward-biased distribution (positive mean)
- Bench2Drive is centered around zero

### 4. Trajectory Y Coordinate Differences
**NavSim**: min=-14.86, max=25.58 meters, mean=0.375, std=1.99  
**Bench2Drive**: min=-39.41, max=36.62 meters, mean=-0.055, std=6.80

- Bench2Drive has 3.4x larger standard deviation
- Range is approximately 2x larger

### 5. Agent Information
- NavSim: agent_states and agent_labels contain actual values
- Bench2Drive: All agent_states and agent_labels are exactly zero

### 6. BEV Semantic Map Classes
- NavSim: Uses classes 0-6
- Bench2Drive: Uses classes 0-4 only

## Normalization Impact Analysis

With current normalization parameters:
```python
"bench2drive": {
    "x": {"offset": 35.0, "scale": 70.0},
    "y": {"offset": 35.0, "scale": 70.0},
    "heading": {"offset": 0.05, "scale": 0.5}
}
```

The heading normalization produces:
- Input range: [-0.110, 0.110]
- Normalized range: [-0.32, -0.12]
- All values map to negative range

## Observations Requiring Investigation

1. **Status Feature**: Why does Bench2Drive have -26.97 as minimum value?
2. **Heading**: Why is the range exactly ±0.110? Is this clipping or different units?
3. **Coordinates**: Why are X values centered at zero in Bench2Drive vs forward-biased in NavSim?
4. **Agent States**: Why are all agent values zero in Bench2Drive?

## Next Investigation Steps

1. **Examine Cache Generation Process**
   - Review Bench2Drive feature/target builders
   - Check data extraction from raw Bench2Drive format
   - Verify unit conversions and coordinate transformations

2. **Compare with Original Data**
   - Load raw Bench2Drive scenarios
   - Extract same fields before caching
   - Identify where differences are introduced

3. **Validate Cached Data**
   - Check if cached values match expected physical ranges
   - Verify coordinate systems and units
   - Ensure proper data field mapping