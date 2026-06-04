# Heart Transplant Gender Stratification

## Background

Heart transplantation is the definitive treatment for end-stage heart failure; however, long-term survival outcomes remain highly variable due to complex interactions among recipient, donor, pharmacologic, and physiologic factors. This variability necessitates robust and interpretable analytical approaches for high-dimensional clinical data.

## Methods

This study analyzed adult heart transplant recipients from the United Network for Organ Sharing (UNOS) registry using a 10-year contemporary cohort to reduce confounding from historical changes in allocation policy and clinical practice.

A structured preprocessing pipeline was implemented, including feature harmonization across clinical, donor, and medication-related variables. Statistical feature screening was applied to remove non-informative predictors. Principal Component Analysis (PCA) was then used for dimensionality reduction, transforming correlated clinical variables into orthogonal components capturing dominant covariance structure. This approach reduced multicollinearity and stabilized downstream modeling.

A gender-stratified design was applied, with male and female recipient cohorts analyzed separately to assess sex-specific differences in risk profiles, donor-recipient interactions, and survival patterns. Survival modeling was conducted using a structured statistical framework to evaluate associations between derived features and post-transplant outcomes.

## Results

The feature reduction pipeline effectively condensed high-dimensional clinical variables into a smaller set of orthogonal components while preserving dominant variance structure. This improved model stability and reduced redundancy among correlated predictors.

Gender-stratified analyses revealed distinct patterns of risk distribution between male and female cohorts, suggesting heterogeneity in donor-recipient interactions and survival-associated clinical profiles that may be obscured in pooled analyses. The integrated framework enabled clearer separation of latent clinical structure and improved interpretability of survival-relevant feature patterns.

## Conclusions

A structured combination of statistical feature screening, PCA-based dimensionality reduction, and gender-stratified survival analysis provides a coherent framework for modeling heterogeneity in post-transplant outcomes. This approach supports more stable and interpretable assessment of risk factors in heart transplant survival modeling.

## Repository Structure

```text
UNOS_GenderStratification/
├── doc/
│   ├── 
│   └── 
├── README.md
├── data/
|   ├── THORACIC_DATA.htm
|   |── THORACIC_DATA.DAT
|   |── THORACIC_FORMATS_FLATFILE.htm
|   |── THORACIC_FORMATS_FLATFILE.DAT
|   └── optn-star-files-data-dictionary.xlsx
├── 
├── figures/
│   ├── 
│   ├── 
│   └── 
├── modules/
│   ├── heart_imputation_diagnose.py
│   ├── heart_plot.py
|   ├── heart_utilities.py
│   └── 
└── notebook/
    ├── A_WranglerPrep.ipynb
    ├── B_WrangleDonorCandidate.ipynb
    ├── C_WrangleExploratoryCandidate.ipynb
    ├── C_WrangleExploratoryDonor.ipynb
    ├── D_WrangleFemaleCandidate.ipynb
    └── 
```
