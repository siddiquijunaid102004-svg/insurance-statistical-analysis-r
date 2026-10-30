# Week 3 Task: Statistical Analysis and Predictive Modeling Using R
# Project: Predicting US Health Insurance Charges
# Beginner-friendly script. Run from the project root folder.
# Required packages: ggplot2, dplyr, tidyr, caret, broom

required_packages <- c("ggplot2", "dplyr", "tidyr", "caret", "broom")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}
invisible(lapply(required_packages, library, character.only = TRUE))

# Create output folders if they do not exist
if (!dir.exists("outputs")) dir.create("outputs")
if (!dir.exists("data")) dir.create("data")

# Load the included public dataset
insurance <- read.csv("data/insurance.csv", stringsAsFactors = FALSE)

# Basic validation and cleaning
cat("\n--- Dataset overview ---\n")
cat("Rows:", nrow(insurance), " Columns:", ncol(insurance), "\n")
print(str(insurance))
print(summary(insurance))
cat("\nMissing values by column:\n")
print(colSums(is.na(insurance)))
cat("Duplicate rows:", sum(duplicated(insurance)), "\n")

# Standardize categorical values and convert to factors
insurance <- insurance %>%
  mutate(
    sex = factor(tolower(trimws(sex))),
    smoker = factor(tolower(trimws(smoker))),
    region = factor(tolower(trimws(region)))
  )

# Remove exact duplicate rows only if present (report the count first)
insurance <- distinct(insurance)

# Save cleaned data
write.csv(insurance, "data/insurance_cleaned.csv", row.names = FALSE)

# Descriptive statistics
numeric_summary <- data.frame(
  variable = c("age", "bmi", "children", "charges"),
  mean = sapply(insurance[c("age", "bmi", "children", "charges")], mean, na.rm = TRUE),
  median = sapply(insurance[c("age", "bmi", "children", "charges")], median, na.rm = TRUE),
  sd = sapply(insurance[c("age", "bmi", "children", "charges")], sd, na.rm = TRUE),
  minimum = sapply(insurance[c("age", "bmi", "children", "charges")], min, na.rm = TRUE),
  maximum = sapply(insurance[c("age", "bmi", "children", "charges")], max, na.rm = TRUE)
)
write.csv(numeric_summary, "outputs/numeric_summary_R.csv", row.names = FALSE)
print(numeric_summary)

# Exploratory visualizations
p1 <- ggplot(insurance, aes(x = charges)) +
  geom_histogram(bins = 35, fill = "#2F6690", color = "white") +
  labs(title = "Distribution of Insurance Charges", x = "Insurance charges (USD)", y = "Number of records") +
  theme_minimal()
ggsave("outputs/R_charges_distribution.png", p1, width = 7, height = 4.5, dpi = 300)

p2 <- ggplot(insurance, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot(outlier.alpha = 0.25) +
  labs(title = "Insurance Charges by Smoking Status", x = "Smoking status", y = "Insurance charges (USD)") +
  theme_minimal() + theme(legend.position = "none")
ggsave("outputs/R_charges_by_smoker.png", p2, width = 7, height = 4.5, dpi = 300)

p3 <- ggplot(insurance, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.55) +
  labs(title = "Age and Insurance Charges by Smoking Status", x = "Age (years)", y = "Insurance charges (USD)") +
  theme_minimal()
ggsave("outputs/R_age_vs_charges.png", p3, width = 7, height = 4.5, dpi = 300)

p4 <- ggplot(insurance, aes(x = bmi, y = charges, color = smoker)) +
  geom_point(alpha = 0.55) +
  labs(title = "BMI and Insurance Charges by Smoking Status", x = "BMI", y = "Insurance charges (USD)") +
  theme_minimal()
ggsave("outputs/R_bmi_vs_charges.png", p4, width = 7, height = 4.5, dpi = 300)

# Hypothesis test:
# H0: mean insurance charges are equal for smokers and non-smokers.
# H1: mean insurance charges differ between smokers and non-smokers.
smoker_test <- t.test(charges ~ smoker, data = insurance, var.equal = FALSE)
cat("\n--- Welch two-sample t-test: charges by smoking status ---\n")
print(smoker_test)
capture.output(smoker_test, file = "outputs/hypothesis_test_R.txt")

smoker_summary <- insurance %>%
  group_by(smoker) %>%
  summarise(n = n(), mean_charges = mean(charges), median_charges = median(charges),
            sd_charges = sd(charges), .groups = "drop")
write.csv(smoker_summary, "outputs/charges_by_smoker_R.csv", row.names = FALSE)
print(smoker_summary)

# Correlation analysis for numeric variables
numeric_data <- insurance %>% select(age, bmi, children, charges)
correlation_matrix <- cor(numeric_data, use = "complete.obs", method = "pearson")
write.csv(correlation_matrix, "outputs/correlation_matrix_R.csv")
cat("\n--- Pearson correlation matrix ---\n")
print(round(correlation_matrix, 3))

# Predictive model: multiple linear regression
# Set seed for reproducibility; hold out 20% of the data for testing.
set.seed(42)
train_rows <- createDataPartition(insurance$charges, p = 0.80, list = FALSE)
train_data <- insurance[train_rows, ]
test_data <- insurance[-train_rows, ]

model <- lm(charges ~ age + sex + bmi + children + smoker + region, data = train_data)
cat("\n--- Regression model summary ---\n")
print(summary(model))
capture.output(summary(model), file = "outputs/model_summary_R.txt")

# Predict on the held-out test set
predictions <- predict(model, newdata = test_data)
residuals_test <- test_data$charges - predictions
mae <- mean(abs(residuals_test))
rmse <- sqrt(mean(residuals_test^2))
r_squared_test <- 1 - sum(residuals_test^2) / sum((test_data$charges - mean(test_data$charges))^2)

metrics <- data.frame(
  metric = c("Test MAE (USD)", "Test RMSE (USD)", "Test R-squared"),
  value = c(mae, rmse, r_squared_test)
)
write.csv(metrics, "outputs/model_metrics_R.csv", row.names = FALSE)
write.csv(data.frame(actual_charges = test_data$charges,
                     predicted_charges = predictions,
                     residual = residuals_test),
          "outputs/test_predictions_R.csv", row.names = FALSE)
print(metrics)

# Five-fold cross-validation using caret
set.seed(42)
cv_control <- trainControl(method = "cv", number = 5)
cv_model <- train(charges ~ age + sex + bmi + children + smoker + region,
                  data = insurance, method = "lm", trControl = cv_control,
                  metric = "RMSE")
cat("\n--- Five-fold cross-validation summary ---\n")
print(cv_model)
capture.output(cv_model, file = "outputs/cross_validation_R.txt")

# Model diagnostic plots
png("outputs/R_model_diagnostics.png", width = 1400, height = 1000, res = 150)
par(mfrow = c(2, 2))
plot(model)
dev.off()

diagnostic_data <- data.frame(predicted = predictions, residual = residuals_test)
p5 <- ggplot(diagnostic_data, aes(x = predicted, y = residual)) +
  geom_point(alpha = 0.6, color = "#2F6690") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residuals vs Predicted Charges", x = "Predicted charges (USD)", y = "Residual (USD)") +
  theme_minimal()
ggsave("outputs/R_residuals_vs_predicted.png", p5, width = 7, height = 4.5, dpi = 300)

p6 <- ggplot(data.frame(actual = test_data$charges, predicted = predictions),
             aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.6, color = "#2F6690") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = "Actual vs Predicted Insurance Charges", x = "Actual charges (USD)", y = "Predicted charges (USD)") +
  theme_minimal()
ggsave("outputs/R_actual_vs_predicted.png", p6, width = 7, height = 4.5, dpi = 300)

# Optional normality check on a sample of residuals; large samples make this test sensitive.
set.seed(42)
resid_sample <- sample(residuals(model), min(5000, length(residuals(model))))
if (length(resid_sample) >= 3 && length(resid_sample) <= 5000) {
  cat("\n--- Shapiro-Wilk residual normality test (interpret cautiously) ---\n")
  print(shapiro.test(resid_sample))
}

cat("\nProject completed. Review the files in the outputs folder.\n")
