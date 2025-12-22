# Data Sources Documentation

## Dataset Overview
- **Date Downloaded:** December 21, 2024
- **Storage:** AWS S3 - `s3://ad-rnaseq-prediction-data/raw/GSE63061/`
- **Status:** Sample data created due to download encoding issues

## Sample Distribution
| Diagnosis | Count | Percentage |
|-----------|-------|------------|
| AD | 145 | 37.3% |
| Control | 134 | 34.4% |
| MCI | 80 | 20.6% |
| MCI_converter | 30 | 7.7% |
| **Total** | **389** | **100%** |

## Clinical Variables
- **Age:** 75.0 ± 8.0 years (range: 50-95)
- **Sex:** Mixed (M/F)
- **MMSE:** Varies by diagnosis
  - Control: ~29
  - MCI: ~25
  - MCI_converter: ~24
  - AD: ~18

## Prediction Targets Defined

### Primary Target: MCI to AD Conversion
- **Total MCI samples:** 110
- **Converters:** 30 (27.3%)
- **Non-converters:** 80 (72.7%)
- **Challenge:** Class imbalance (1:2.67 ratio)
- **Strategy:** Use balanced accuracy, SMOTE, or class weights

### Secondary Target: Disease State Classification
- **Classes:** Control (134), MCI (80), AD (145)
- **Use case:** Diagnostic support tool

### Additional Target: MMSE Score Prediction
- **Type:** Regression
- **Range:** 10-30
- **Use case:** Cognitive trajectory prediction