# %% Multivariate Analysis for Outcome 2
# This script conducts multivariate logistic regression analysis to identify
# the most impactful variables for outcome 2, controlling for centro_studi

# Configuration: Set to None for all centers, or specify center name
SELECTED_CENTER = None  # Options: None, "San Raffaele", "Casilino"

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pandas as pd
from tesi_marghe.utils.polars_utils import print_shape
from tesi_marghe.utils.log_reg import (
    run_multivariate_logistic_analysis,
    run_complete_logistic_analysis,
)

import logging

logger = logging.getLogger(__name__)

# %% Load data
data = pl.read_excel(
    "/Users/valensisecarlo/Personal_Projects/tesi_marghe/data/2025-06-28 1119_dati_multicentro.xlsx"
)

# Apply center filter if specified
if SELECTED_CENTER is not None:
    print(f"Filtering to center: {SELECTED_CENTER}")
    data = data.filter(pl.col("Centro_studi") == SELECTED_CENTER)
else:
    print("Using all centers")

data.columns = (
    pl.DataFrame(data.columns)
    .with_columns(lower_camel=pl.col("column_0").str.to_lowercase().str.replace_all(" ", "_"))
    .select("lower_camel")
    .to_series()
    .to_list()
)

print(f"Data shape: {data.shape}")
print(f"Outcome 2 distribution:")
print(data.group_by("outcome_2").agg(pl.len().alias("count")))

# Check if we have centro_studi variation
if SELECTED_CENTER is None:
    print("Center distribution:")
    print(data.group_by("centro_studi").agg(pl.len().alias("count")))
else:
    print(f"Analysis restricted to: {SELECTED_CENTER}")
    print("Note: Centro_studi will not be included as control variable (constant)")

# %% Define variable groups based on 1_analisi_descrittiva.py

# Demographics
demographic_vars = ["bmi", "bmi_pregravidico", "età", "peso_neonato"]

# Ultrasound variables
ultrasound_vars = ["cpr", "afi", "ca", "pfs", "pi_ut_medio", "diam_medio", "vel_media", "qvo"]

# Hemodynamic variables
hemodynamic_vars = ["pasc", "padc", "svr", "co", "ci", "tfc", "ino", "pkr", "sv"]

# CTG variables (normalized by duration)
ctg_vars = [
    # "maf/h",
    "fhr",
    "accelerazioni",
    "decelerazioni",
    "stv",
    "durata_episodi_alta_variazione",
    "durata_episodi_bassa_variazione",
]


# All variables of interest (excluding centro_studi which is always a control)
all_vars_of_interest = demographic_vars + ultrasound_vars + hemodynamic_vars + ctg_vars

print(f"Total variables to analyze: {len(all_vars_of_interest)}")
print(f"Demographic: {len(demographic_vars)}")
print(f"Ultrasound: {len(ultrasound_vars)}")
print(f"Hemodynamic: {len(hemodynamic_vars)}")
print(f"CTG: {len(ctg_vars)}")

# %% Step 1: Univariate screening
# Run univariate analysis for each variable to identify candidates for multivariate model

print("\n" + "=" * 80)
print("STEP 1: UNIVARIATE SCREENING")
print("=" * 80)

univariate_results = []

for var in all_vars_of_interest:
    print(f"\nAnalyzing: {var}")

    try:
        # Determine control variables based on center selection
        if SELECTED_CENTER is None:
            control_vars = ["centro_studi"]  # Use centro_studi as control
        else:
            control_vars = []  # No control variables when single center

        # For CTG variables, we need to normalize by duration first
        if var in ctg_vars:
            # Create normalized variable
            temp_data = data.with_columns(pl.col(var) / pl.col("durata_ctg")).rename(
                {var: f"{var}_norm"}
            )

            results = run_complete_logistic_analysis(
                data=temp_data,
                predictor_cols=[f"{var}_norm"],
                control_cols=control_vars,
                target_col="outcome_2",
                analysis_name=f"{var}_norm vs Outcome 2"
                + (f" ({SELECTED_CENTER})" if SELECTED_CENTER else ""),
                print_results=False,
                plot_results=False,
            )
        else:
            results = run_complete_logistic_analysis(
                data=data,
                predictor_cols=[var],
                control_cols=control_vars,
                target_col="outcome_2",
                analysis_name=f"{var} vs Outcome 2"
                + (f" ({SELECTED_CENTER})" if SELECTED_CENTER else ""),
                print_results=False,
                plot_results=False,
            )

        # Extract key results
        main_result = results["odds_ratios"].row(0, named=True)
        summary = results["summary_results"].row(0, named=True)

        univariate_results.append(
            {
                "Variable": var if var not in ctg_vars else f"{var}_norm",
                "OR": main_result["Odds_Ratio"],
                "OR_CI_Lower": main_result["OR_CI_Lower"],
                "OR_CI_Upper": main_result["OR_CI_Upper"],
                "AUC": summary["AUC"],
                "AUC_p_value": summary["AUC_p_value"],
                "Variable_Type": "CTG"
                if var in ctg_vars
                else (
                    "Demographic"
                    if var in demographic_vars
                    else ("Ultrasound" if var in ultrasound_vars else "Hemodynamic")
                ),
                "Center": SELECTED_CENTER if SELECTED_CENTER else "All",
            }
        )

        print(
            f"  OR: {main_result['Odds_Ratio']:.3f} (95% CI: {main_result['OR_CI_Lower']:.3f}-{main_result['OR_CI_Upper']:.3f})"
        )
        print(f"  AUC: {summary['AUC']:.3f} (p={summary['AUC_p_value']:.4g})")

    except Exception as e:
        print(f"  Error analyzing {var}: {e}")
        univariate_results.append(
            {
                "Variable": var if var not in ctg_vars else f"{var}_norm",
                "OR": np.nan,
                "OR_CI_Lower": np.nan,
                "OR_CI_Upper": np.nan,
                "AUC": np.nan,
                "AUC_p_value": np.nan,
                "Variable_Type": "CTG"
                if var in ctg_vars
                else (
                    "Demographic"
                    if var in demographic_vars
                    else ("Ultrasound" if var in ultrasound_vars else "Hemodynamic")
                ),
                "Center": SELECTED_CENTER if SELECTED_CENTER else "All",
            }
        )

# Create univariate results table
univariate_df = pl.DataFrame(univariate_results)
print(f"\nUnivariate screening completed for {len(univariate_results)} variables")

# %% Step 2: Select variables for multivariate analysis
# Criteria: p < 0.1 for AUC and/or OR significantly different from 1

print("\n" + "=" * 80)
print("STEP 2: VARIABLE SELECTION FOR MULTIVARIATE ANALYSIS")
print("=" * 80)

# Filter variables based on significance
significant_vars = univariate_df.filter(
    (pl.col("AUC_p_value") < 0.1)  # AUC significantly different from 0.5
    | (
        (pl.col("OR_CI_Lower") > 1) | (pl.col("OR_CI_Upper") < 1)
    )  # OR significantly different from 1
).sort("AUC_p_value")

print(f"Variables meeting significance criteria (p < 0.1 or OR ≠ 1): {len(significant_vars)}")
print(significant_vars.select(["Variable", "OR", "AUC", "AUC_p_value", "Variable_Type"]))

# Alternative: Top variables by AUC
top_auc_vars = (
    univariate_df.filter(pl.col("AUC").is_not_null()).sort("AUC", descending=True).head(10)
)

print(f"\nTop 10 variables by AUC:")
print(top_auc_vars.select(["Variable", "OR", "AUC", "AUC_p_value", "Variable_Type"]))

# %% Step 3: Multivariate analysis with selected variables

print("\n" + "=" * 80)
print("STEP 3: MULTIVARIATE ANALYSIS")
print("=" * 80)

# Prepare data for multivariate analysis
# We'll use the significant variables plus a few top performers

# Select variables for multivariate model
multivariate_candidates = significant_vars["Variable"].to_list()

multivariate_candidates = [
    "cpr",
    "svr",
    "afi",
    # "cqvo",
    # "pi_ut_medio",
    "pregressi_ps",
    "partoanalgesia",
    "bmi_pregravidico",
]

# If we have too many variables, limit to top performers
if len(multivariate_candidates) > 10:
    # Take top 8 by AUC
    multivariate_candidates = top_auc_vars.head(8)["Variable"].to_list()

print(f"Variables selected for multivariate analysis: {multivariate_candidates}")

# Prepare data with normalized CTG variables
multivariate_data = data.clone()

# Add normalized CTG variables
for var in ctg_vars:
    if f"{var}_norm" in multivariate_candidates:
        multivariate_data = multivariate_data.with_columns(
            pl.col(var) / pl.col("durata_ctg")
        ).rename({var: f"{var}_norm"})

# Determine control variables for multivariate analysis
if SELECTED_CENTER is None:
    control_vars = ["centro_studi"]  # Use centro_studi as control
    analysis_title = "Multivariate Model vs Outcome 2 (All Centers)"
else:
    control_vars = []  # No control variables when single center
    analysis_title = f"Multivariate Model vs Outcome 2 ({SELECTED_CENTER})"

# Run multivariate analysis
try:
    multivariate_results = run_multivariate_logistic_analysis(
        data=multivariate_data,
        predictor_cols=multivariate_candidates,
        control_cols=control_vars,
        target_col="outcome_2",
        analysis_name=analysis_title,
        print_results=True,
        plot_results=True,
    )

    print("\nMultivariate analysis completed successfully!")

except Exception as e:
    print(f"Error in multivariate analysis: {e}")
    # Try with fewer variables
    print("Trying with fewer variables...")

    # Take only top 5 variables
    multivariate_candidates = top_auc_vars.head(5)["Variable"].to_list()
    print(f"Retrying with: {multivariate_candidates}")

    multivariate_results = run_multivariate_logistic_analysis(
        data=multivariate_data,
        predictor_cols=multivariate_candidates,
        control_cols=control_vars,
        target_col="outcome_2",
        analysis_name=f"Multivariate Model (Top 5) vs Outcome 2"
        + (f" ({SELECTED_CENTER})" if SELECTED_CENTER else ""),
        print_results=True,
        plot_results=True,
    )

# %% Step 4: Variable importance analysis

print("\n" + "=" * 80)
print("STEP 4: VARIABLE IMPORTANCE ANALYSIS")
print("=" * 80)

# Extract odds ratios from multivariate model
multivariate_or = multivariate_results["odds_ratios"]

# Filter out centro_studi (control variable, not clinically relevant)
multivariate_or_clinical = multivariate_or.filter(~pl.col("Variable").str.contains("centro_studi"))

print("Adjusted Odds Ratios (multivariate model) - Clinical Variables Only:")
for row in multivariate_or_clinical.iter_rows(named=True):
    print(f"{row['Variable']}: OR = {row['Odds_Ratio']:.3f}", end="")
    if not np.isnan(row["OR_CI_Lower"]):
        print(f" (95% CI: {row['OR_CI_Lower']:.3f}-{row['OR_CI_Upper']:.3f})")
    else:
        print(" (CI: Not available)")

# Create variable importance plot (clinical variables only)
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by absolute log OR
multivariate_or_sorted = multivariate_or_clinical.with_columns(
    pl.col("Odds_Ratio").log().abs().alias("abs_log_or")
).sort("abs_log_or", descending=True)

y_positions = np.arange(len(multivariate_or_sorted))
or_values = multivariate_or_sorted["Odds_Ratio"].to_numpy()
ci_lower = multivariate_or_sorted["OR_CI_Lower"].to_numpy()
ci_upper = multivariate_or_sorted["OR_CI_Upper"].to_numpy()
variables = multivariate_or_sorted["Variable"].to_numpy()
variables_eng = [
    "Previous Spontaneous Deliveries",
    "SVR",
    "CPR",
    "AFI",
    "Analgesia",
    "Pre-Pregnancy BMI",
]

# Plot error bars
for i, (or_val, ci_l, ci_u, var) in enumerate(zip(or_values, ci_lower, ci_upper, variables)):
    if not np.isnan(ci_l):
        ax.errorbar(
            or_val,
            i,
            xerr=[[or_val - ci_l], [ci_u - or_val]],
            fmt="o",
            capsize=5,
            capthick=2,
            elinewidth=2,
            markersize=8,
            color="blue",
        )
    else:
        ax.scatter(or_val, i, s=100, color="blue", marker="o")

    # Add variable name
    ax.text(or_val, i + 0.1, variables_eng[i], ha="left", va="bottom", fontsize=10)

ax.axvline(x=1, color="red", linestyle="--", alpha=0.7, label="No effect (OR=1)")
ax.set_yticks(y_positions)

ax.set_yticklabels(variables_eng)
ax.set_xlabel("Adjusted Odds Ratio")
ax.set_title("Variable Importance: Multivariate Model vs Outcome 2")
ax.grid(True, alpha=0.3, axis="x")
ax.set_xscale("log")
ax.legend()

plt.tight_layout()
plt.show()

# %% Step 5: Model comparison and summary

print("\n" + "=" * 80)
print("STEP 5: MODEL COMPARISON AND SUMMARY")
print("=" * 80)

# Compare univariate vs multivariate performance
print("Model Performance Comparison:")
print(f"Multivariate Model AUC: {multivariate_results['roc_metrics']['auc_score']:.3f}")
print(f"Multivariate Model p-value: {multivariate_results['roc_metrics']['p_value']:.4g}")

# Find best univariate model
best_univariate = (
    univariate_df.filter(pl.col("AUC").is_not_null())
    .sort("AUC", descending=True)
    .row(0, named=True)
)

print(f"Best Univariate Model: {best_univariate['Variable']}")
print(f"Best Univariate AUC: {best_univariate['AUC']:.3f}")
print(f"Best Univariate p-value: {best_univariate['AUC_p_value']:.4g}")

# Improvement
auc_improvement = multivariate_results["roc_metrics"]["auc_score"] - best_univariate["AUC"]
print(f"AUC Improvement: {auc_improvement:.3f}")

# Summary table
summary_table = pl.DataFrame(
    {
        "Model_Type": ["Best Univariate", "Multivariate"],
        "Best_Variable": [best_univariate["Variable"], "Multiple"],
        "AUC": [best_univariate["AUC"], multivariate_results["roc_metrics"]["auc_score"]],
        "AUC_p_value": [
            best_univariate["AUC_p_value"],
            multivariate_results["roc_metrics"]["p_value"],
        ],
        "N_Variables": [1, len(multivariate_candidates)],
    }
)

print("\nSummary Table:")
print(summary_table)

# %% Step 6: Export results

print("\n" + "=" * 80)
print("STEP 6: EXPORTING RESULTS")
print("=" * 80)

# Create file suffix based on center selection
file_suffix = f"_{SELECTED_CENTER.lower().replace(' ', '_')}" if SELECTED_CENTER else "_all_centers"

# Export univariate results
univariate_df.write_csv(f"univariate_screening_outcome2{file_suffix}.csv")
print(f"Univariate results exported to: univariate_screening_outcome2{file_suffix}.csv")

# Export multivariate results
multivariate_results["odds_ratios"].write_csv(f"multivariate_results_outcome2{file_suffix}.csv")
print(f"Multivariate results exported to: multivariate_results_outcome2{file_suffix}.csv")

# Export summary
summary_table.write_csv(f"model_comparison_outcome2{file_suffix}.csv")
print(f"Model comparison exported to: model_comparison_outcome2{file_suffix}.csv")

print("\nAnalysis completed! Key findings:")
print(f"1. {len(significant_vars)} variables showed significant association with outcome 2")
print(f"2. Multivariate model includes {len(multivariate_candidates)} variables")
print(f"3. Multivariate AUC: {multivariate_results['roc_metrics']['auc_score']:.3f}")
print(f"4. Most important variables identified in multivariate analysis")
print(f"5. Analysis scope: {SELECTED_CENTER if SELECTED_CENTER else 'All centers'}")

# %% Additional analysis: Stepwise selection (if needed)

print("\n" + "=" * 80)
print("ADDITIONAL: STEPWISE VARIABLE SELECTION")
print("=" * 80)

# This section can be used for more sophisticated variable selection
# For now, we'll show how to do forward selection manually

print("Manual forward selection approach:")
print("1. Start with the variable with highest univariate AUC")
print("2. Add variables one by one, checking if they improve the model")
print("3. Stop when adding variables doesn't significantly improve performance")

# Example: Start with best univariate variable
best_var = best_univariate["Variable"]
print(f"\nStarting with: {best_var}")

# You can implement stepwise selection here if needed
# For now, we've used the multivariate model with pre-selected variables

print("\nAnalysis script completed successfully!")
