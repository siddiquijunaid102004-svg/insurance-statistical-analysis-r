# Statistical Analysis and Predictive Modeling Using R
## Project: Predicting US Health Insurance Charges

This project explores a public health insurance dataset and builds a multiple linear regression model to estimate insurance charges from age, sex, BMI, number of children, smoking status, and region.

### Dataset
- File: `data/insurance.csv`
- Records: 1,338
- Columns: 7
- Target variable: `charges` (USD)
- Public source: Kaggle's Medical Cost Personal Dataset. See the source reference in the report.

### Folder structure
- `data/insurance.csv` — original uploaded dataset copy
- `data/insurance_cleaned.csv` — cleaned data copy created for this submission
- `scripts/insurance_analysis.R` — complete R analysis and modeling script
- `outputs/` — charts, summary tables, hypothesis test and model outputs
- `reports/Insurance_Statistical_Analysis_Report.docx` — main report

### Run the project
1. Install R from CRAN and open RStudio or VS Code with an R extension.
2. Open the folder `insurance_predictive_modeling_project`.
3. Set the working directory to this project root.
4. Run `source("scripts/insurance_analysis.R")` in the R console.
5. If prompted, allow the script to install the required packages.
6. Review generated files in `outputs/`.

### Statistical approach
- Data quality checks: missing values, duplicates, types and summaries.
- EDA: charge distribution, charge comparisons by smoking status, age and BMI scatterplots.
- Hypothesis test: Welch two-sample t-test comparing charges by smoking status.
- Prediction: multiple linear regression.
- Validation: 80/20 holdout and five-fold cross-validation.
- Metrics: MAE, RMSE and test-set R-squared.
- Diagnostics: residual plots and standard regression diagnostic plots.

### Key findings (from analysis of the supplied file)
- Dataset has 1,338 records and 7 variables.
- Missing values: 0; exact duplicate rows: 1.
- Mean insurance charge: $13,270.42; median: $9,382.03.
- Average charge among smokers: $32,050.23; among non-smokers: $8,434.27.
- The model's holdout metrics are recorded in `outputs/model_metrics.csv`. The R script recomputes these in R.

### Important limitation
This is an educational analysis of an anonymized historical dataset, not a clinical or insurance-pricing recommendation. Associations are not proof of causation, and the model may underpredict unusually high charges.
