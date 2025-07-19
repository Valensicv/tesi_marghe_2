# %% loadings|


%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.polars
import polars as pl
import polars_ds as pds
from tesi_marghe.utils.polars_utils import print_shape
from tesi_marghe.utils.waterfall import create_waterfall_figure

from tesi_marghe.utils.ttests_utils import compute_group_comparison_table

# Imports for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

import logging
logger = logging.getLogger(__name__)

# %%
hvplot.extension('matplotlib')

data = pl.read_excel("/Users/valensisecarlo/Personal_Projects/tesi_marghe/data/2025-06-28 1119_dati_multicentro.xlsx")

data.columns = pl.DataFrame(data.columns).with_columns(
    lower_camel = pl.col("column_0").str.to_lowercase().str.replace_all(" ", "_")
).select('lower_camel').to_series().to_list()
# %% perimetro - numero pazienti

perimeter = (
    data
    .pipe(print_shape, label='before drop nulls')
    .drop_nulls("cognome")
    .pipe(print_shape, label='after drop nulls')
    .group_by("centro_studi").agg(
    pl.len().alias("numero_pazienti"),
).sort("numero_pazienti", descending=True)
)

perimeter.write_clipboard()



centro_studi = perimeter['centro_studi'].to_list() + ["Total"]
numero_pazienti = perimeter['numero_pazienti'].to_list() + [perimeter['numero_pazienti'].sum()]

fig = create_waterfall_figure(
    name = "Centro Studi",
    orientation = "v",
    measure = ["relative", "relative", "total"],
    x = centro_studi,
    y = numero_pazienti,
    text = numero_pazienti,
    textposition = "outside",
    connector = {"line":{"color":"rgb(63, 63, 63)"}},

)

fig.update_layout(title="Number of patients per center")
fig.show()


# %% perimetro - numero pazienti per outcome 1

perimeter = (
    data
    .pipe(print_shape, label='before drop nulls')
    .drop_nulls("cognome")
    .pipe(print_shape, label='after drop nulls')
    .group_by(["centro_studi",'outcome_1']).agg(
    pl.len().alias("numero_pazienti"),
).sort("centro_studi",'outcome_1', descending=True).with_columns(
    tot_paz = pl.col("numero_pazienti").sum().over("centro_studi")
).with_columns(
    percentuale = pl.col("numero_pazienti") / pl.col("tot_paz")
)

)
perimeter.write_clipboard()

perimeter


# %% perimetro - numero pazienti per outcome 2


perimeter = (
    data
    .pipe(print_shape, label='before drop nulls')
    .drop_nulls("cognome")
    .pipe(print_shape, label='after drop nulls')
    .group_by(["centro_studi",'outcome_2']).agg(
    pl.len().alias("numero_pazienti"),
).sort("centro_studi",'outcome_2', descending=True).with_columns(
    tot_paz = pl.col("numero_pazienti").sum().over("centro_studi")
).with_columns(
    percentuale = pl.col("numero_pazienti") / pl.col("tot_paz")
)

)
perimeter.write_clipboard()

perimeter

# %% istogrammi variabili demografiche
%matplotlib inline
import hvplot.pandas

variabili = [
# "altezza",
# "peso",
# "peso_pregravidico",
"bmi",
"bmi_pregravidico",
"età",
# "eg_week",
# "eg_day",
# "gravida",
# "parità",
# "pregressi_as",
# "pregressi_ps",
# "pregressi_tc",
# "pregressi_ps_pretermine",
# "pregressi_tc_pretermine",
# "fumo",
"peso_neonato",
'centro_studi'
]

hv.extension('matplotlib')
plots = []

for var in variabili:
    if var != 'centro_studi':
        plots.append(
            data
            .select(variabili)
            .hvplot.hist(var,by='centro_studi',height = 70, width = 70, alpha= 0.5, title=var).opts(show_legend = len(plots)==len(variabili)-2)
            # .hist(pl.all().exclude('centro_studi'),by='centro_studi')
        )
hv.Layout(plots).cols(4).opts(vspace = 0.3,hspace = 0.2)

result_table = compute_group_comparison_table(data.select(variabili))
result_table

# %% variabili ecografiche - istogrammi

variabili= [
# 'piao',
# 'piao_pc',
# 'piacm',
# 'piacm_pc',
'cpr',
# 'cpr_pc',
'afi',
'ca',
# 'ca_pc',
'pfs',
# 'pfs_pc',
# 'pi_a_ut_dx',
# 'pi_a_ut_sx',
'pi_ut_medio',
# 'diam1',
# 'diam2',
# 'diam3',
# 'vel1',
# 'vel2',
# 'vel3',
'diam_medio',
'vel_media',
'qvo',
# 'cqvo',
'centro_studi'
]


# f = sns.pairplot(
#     data
#     .select(variabili)
#     .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
#     .to_pandas(), hue="centro_studi" )
# plt.show()

hv.extension('matplotlib')
plots = []

for var in variabili:
    if var != 'centro_studi':
        plots.append(
            data
            .select(variabili)
            .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
            .hvplot.hist(var,by='centro_studi',height = 70, width = 70, alpha= 0.5, title=var).opts(show_legend = len(plots)==len(variabili)-2)
            # .hist(pl.all().exclude('centro_studi'),by='centro_studi')
        )
hv.Layout(plots).cols(4).opts(vspace = 0.3,hspace = 0.2)


# Example usage:
result_table = compute_group_comparison_table(data.select(variabili))

conclusioni = """
Across the eight vascular-metric comparisons between the San Raffaele and Casilino cohorts we first ran two-sample t-tests.  Five variables—vel_media, afi, cpr, diam_medio, qvo—showed p-values below the Bonferroni-corrected threshold of 0.006 (so family-wise α≈0.05 is respected), while ca, pi_ut_medio, pfs did not reach that stricter bar.  To gauge practical relevance we transformed each t into Hedges g: vel_media (g≈–1.04) and afi (g≈ 0.77) are large / medium-large effects, indicating velocities are markedly lower and amniotic-fluid indices clearly higher, respectively, at Casilino versus San Raffaele (sign is negative when Casilino > San Raffaele).  cpr registers a solid medium effect (g≈–0.50), whereas diam_medio and qvo sit in the small-to-medium band (g≈ 0.42 and –0.44).  The remaining variables cluster below |g| = 0.25, suggesting clinically minor or uncertain gaps even where raw p suggested “significance.”  Ninety-five-percent CIs accompany every g; for the five Bonferroni-survivors the intervals stay outside ±0.20, reinforcing that those differences are unlikely to be negligible.  Overall, statistical evidence and effect-size magnitude align on a handful of variables that warrant clinical attention, while the rest are best viewed as either false alarms from multiple testing or changes too small to matter in practice.
"""



# %% variabili emodinamiche  - istogrammi
%matplotlib inline
variabili= [
################## 
'pasc',
'padc',
# 'pasp',
# 'padp',
# 'sai',
# 'hr',
'svr',
# 'svri',
# 'bsa',
'co',
'ci',
'tfc',
'ino',
'pkr',
'sv',
# 'svv',
# 'do2',
# 'do2_calc',
# 'do2i',
# 'spo2',
# 'hb',
'centro_studi'
]

f = sns.pairplot(
    data
    .select(variabili)
    .to_pandas(), hue="centro_studi" )
plt.show()


hv.extension('matplotlib')
plots = []

for var in variabili:
    if var != 'centro_studi':
        plots.append(
            data
            .select(variabili)
            # .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
            .hvplot.hist(var,by='centro_studi',height = 70, width = 70, alpha= 0.5, title=var).opts(show_legend = len(plots)==len(variabili)-2)
            # .hist(pl.all().exclude('centro_studi'),by='centro_studi')
        )
hv.Layout(plots).cols(5).opts(vspace = 0.3,hspace = 0.2)


# Example usage:
result_table = compute_group_comparison_table(data.select(variabili))
result_table



# %% variabili bassa variazione - istogrammi

%matplotlib inline
variabili = [
"durata_ctg",
"maf/h",
"fhr",
"accelerazioni",
"decelerazioni",
"stv",
"durata_episodi_alta_variazione",
"durata_episodi_bassa_variazione",
'centro_studi'
]


f = sns.pairplot(
    data
    .select(variabili)
    .with_columns(
        pl.all().exclude('centro_studi') / pl.col('durata_ctg')
    )
    .to_pandas(), hue="centro_studi" )
plt.show()


hv.extension('matplotlib')
plots = []

for var in variabili:
    if var != 'centro_studi':
        plots.append(
            data
            .select(variabili)
            .with_columns(
        pl.all().exclude('centro_studi') / pl.col('durata_ctg')
    )
            # .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
            .hvplot.hist(var,by='centro_studi',height = 70, width = 70, alpha= 0.5, title=var).opts(show_legend = len(plots)==len(variabili)-2)
            # .hist(pl.all().exclude('centro_studi'),by='centro_studi')
        )
hv.Layout(plots).cols(5).opts(vspace = 0.3,hspace = 0.2,shared_axes=False)


# Example usage:
result_table = compute_group_comparison_table(data.select(variabili)        .with_columns(
        pl.all().exclude('centro_studi') / pl.col('durata_ctg')
    ))
result_table


# %% Correlazioni emodinamiche - ecografiche
%matplotlib inline

variabili= [
# 'cpr',
# 'afi',
# 'ca',
# 'pfs',
# 'pi_ut_medio',

# 'diam_medio',
'vel_media',
# 'qvo',
# -----------------
# 'pasc',
# 'padc',
# 'svr',

'co',
# 'ci',
# 'tfc',
# 'ino',
# 'pkr',
# 'sv',

'centro_studi'
]

f = sns.pairplot(
    data
    .select(variabili)
    # .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
    .to_pandas(), hue="centro_studi" , plot_kws = {'alpha':0.1} )
plt.show()



# %% correlazioni full db
(
    data
    .select(variabili)
    .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
    
    .filter(~pl.any_horizontal(pl.all().is_null()))
    .select(pl.all().exclude('centro_studi'))
    .corr()
    .with_columns(var=pl.Series(variabili[:-1]))
    .unpivot(index='var')
    .filter(pl.col.value.lt(0.999))
    
    .with_columns(
        pl.concat_list(pl.col.var, pl.col.variable).list.sort().alias("var_variable")
    )
    .unique(subset=['var_variable'])
    .drop('var_variable')
    .sort(pl.col.value.abs(), descending=True)
)

# %% correlazioni san raffaele - casilino

corr_sr = (
    data
    .select(variabili)
    .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
    
    .filter(~pl.any_horizontal(pl.all().is_null()))
    .filter(pl.col.centro_studi == 'San Raffaele')
    .select(pl.all().exclude('centro_studi'))
    .corr()
    .with_columns(var=pl.Series(variabili[:-1]))
    .unpivot(index='var')
    .filter(pl.col.value.lt(0.999))
    
    .with_columns(
        pl.concat_list(pl.col.var, pl.col.variable).list.sort().alias("var_variable")
    )
    .unique(subset=['var_variable'])
    .drop('var_variable')
    .sort(pl.col.value.abs(), descending=True)
)


corr_cas = (
    data
    .select(variabili)
    .filter(pl.col.pi_ut_medio.lt(pl.col.pi_ut_medio.quantile(.99)))
    .filter(~pl.any_horizontal(pl.all().is_null()))
    .filter(pl.col.centro_studi == 'Casilino')
    .select(pl.all().exclude('centro_studi'))
    .corr()
    .with_columns(var=pl.Series(variabili[:-1]))
    .unpivot(index='var')
    .filter(pl.col.value.lt(0.999))
    
    .with_columns(
        pl.concat_list(pl.col.var, pl.col.variable).list.sort().alias("var_variable")
    )
    .unique(subset=['var_variable'])
    .drop('var_variable')
    .sort(pl.col.value.abs(), descending=True)
)


join_corr = (
    corr_sr
    .join(corr_cas , on = ['var','variable'], how = 'outer')
    .rename({'value':'San Raffaele','value_right':'Casilino'})
    .drop('var_right','variable_right')
    .with_columns(
        pl.concat_list(pl.col.var, pl.col.variable).list.sort().alias("var_variable")
    )
    .unique(subset=['var_variable'])
    .drop('var_variable')
    .with_columns(
        delta = pl.col('San Raffaele') - pl.col('Casilino')
    )
    .sort(pl.col.delta.abs(), descending=True)
)


join_corr.filter(variable = 'pfs',var = 'qvo')



# %% 

variabili= [
# 'cpr',
'afi',
# 'ca',
# 'pfs',
# 'pi_ut_medio',
# 'diam_medio',
# 'vel_media',
# 'qvo',
# -----------------
# 'pasc',
# 'padc',
# 'svr',
# 'co',
# 'ci',
# 'tfc',
# 'ino',
# 'pkr',
# 'sv',
'centro_studi',
'outcome_1',
]






# %% ROC Curve Analysis no model
%matplotlib inline

PREDITTORE = 'afi'
OUTCOME = 'outcome_1'
OSPEDALE = 'San Raffaele'



sub_df = data.filter(pl.col.centro_studi == OSPEDALE).select(PREDITTORE,OUTCOME).drop_nulls()

outcome = sub_df[OUTCOME]
predictor = -sub_df[PREDITTORE]


from scipy.stats import mannwhitneyu, norm
import numpy as np

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(outcome, predictor)
auc_score = roc_auc_score(outcome, predictor)


term1 = test_afi_casilino.filter(pl.col(OUTCOME)==1)[PREDITTORE]
term2 = test_afi_casilino.filter(pl.col(OUTCOME)==0)[PREDITTORE]


u_stat, p_two_sided = mannwhitneyu(term1, term2, alternative='two-sided')
# Questo p_two_sided corrisponde a H0: AUC = 0.5



distances = np.sqrt((1 - tpr)**2 + fpr**2)

# Trova l'indice del punto con distanza minima
optimal_idx = np.argmin(distances)

# Il punto ottimo
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]
optimal_threshold = thresholds[optimal_idx]

print(f"Punto ottimo:")
print(f"FPR (1 - Specificità): {optimal_fpr:.4f}")
print(f"TPR (Sensibilità): {optimal_tpr:.4f}")
print(f"Threshold: {optimal_threshold:.4f}")
print(f"Distanza dall'angolo (0,1): {distances[optimal_idx]:.4f}")
print(f"AUC={auc_score:.3f}, p={p_two_sided:.4g}")

# Create ROC curve plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
# ROC Curve
ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
ax1.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', alpha=0.7)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate (1 - Specificity)')
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_title(f'ROC Curve: {PREDITTORE} (dati raw)')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# %%

from tesi_marghe.utils.log_reg import run_complete_logistic_analysis

# %% Logistic Regression Analysis: AFI & Centro Studi vs Outcome 1

# Analisi con AFI come predittore principale e Centro Studi come controllo
results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['afi'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="AFI (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# Odds ratios


# Metriche ROC
print(f"AUC: {results_afi['roc_metrics']['auc_score']:.3f}")

# Risultati di sintesi per export
print(results_afi['summary_results'])

# Tabella dati ROC
from tesi_marghe.utils.log_reg import create_roc_data_table
roc_table = create_roc_data_table(results_afi['roc_metrics'])
print(roc_table.head(10))

# %% svr - casilino


results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['svr'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="SVR (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# Odds ratios


# # Metriche ROC
# print(f"AUC: {results_afi['roc_metrics']['auc_score']:.3f}")

# # Risultati di sintesi per export
# print(results_afi['summary_results'])

# # Tabella dati ROC
# from tesi_marghe.utils.log_reg import create_roc_data_table
# roc_table = create_roc_data_table(results_afi['roc_metrics'])
# print(roc_table.head(10))


# %% Esempio di altre analisi che puoi fare

# Analisi con un altro predittore
# results_altra_var = run_complete_logistic_analysis(
#     data=data,
#     predictor_cols=['altra_variabile'],
#     control_cols=['centro_studi'],
#     target_col='outcome_1',
#     analysis_name="Altra Variabile (adjusted for Centro Studi) vs Outcome 1"
# )

# Analisi con più predittori
# results_multiple = run_complete_logistic_analysis(
#     data=data,
#     predictor_cols=['afi', 'altra_variabile'],
#     control_cols=['centro_studi'],
#     target_col='outcome_1',
#     analysis_name="Multiple Predictors vs Outcome 1"
# )

# %% Accesso ai risultati





# %% Logistic Regression Analysis: AFI & Centro Studi vs Outcome 1

# Select variables for logistic regression
# regression_vars = ['afi', 'centro_studi', 'outcome_1']
regression_vars = ['afi', 'centro_studi', 'outcome_1']

# Prepare data for logistic regression
regression_data = (
    data
    .select(regression_vars)
    .drop_nulls()
    .pipe(print_shape, label='Regression data after dropping nulls')
)

print("Data summary:")
print(regression_data.describe())
print("\nOutcome distribution:")
print(regression_data.group_by('outcome_1').agg(pl.len().alias('count')))
print("\nCentro Studi distribution:")
print(regression_data.group_by('centro_studi').agg(pl.len().alias('count')))

# Convert to pandas for sklearn
regression_df = regression_data.to_pandas()

# Encode categorical variable (centro_studi) if it's not already numeric
le = LabelEncoder()
if regression_df['centro_studi'].dtype == 'object':
    regression_df['centro_studi_encoded'] = le.fit_transform(regression_df['centro_studi'])
    predictor_cols = ['afi', 'centro_studi_encoded']
    centro_labels = dict(zip(le.transform(le.classes_), le.classes_))
    print(f"\nCentro Studi encoding: {centro_labels}")
else:
    predictor_cols = ['afi', 'centro_studi']

# Prepare features and target
X = regression_df[predictor_cols]
y = regression_df['outcome_1']

# Standardize features for better interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_scaled, y)

# Calculate odds ratios
# For standardized coefficients, OR = exp(coef)
coefficients = log_reg.coef_[0]
odds_ratios = np.exp(coefficients)

# Calculate confidence intervals using the SCALED design matrix
# This handles the case where centro_studi might be constant (e.g., only Casilino)
try:
    # Use scaled design matrix for proper variance calculation
    X_scaled_df = pl.DataFrame(X_scaled, schema=predictor_cols)
    
    # Add intercept column for proper variance calculation
    X_with_intercept = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
    
    # Calculate variance-covariance matrix
    from scipy import stats
    import scipy.linalg as linalg
    
    # Get predicted probabilities
    y_pred_proba = log_reg.predict_proba(X_scaled)[:, 1]
    
    # Calculate residuals
    residuals = y - y_pred_proba
    
    # Calculate variance-covariance matrix
    XtX_inv = linalg.inv(X_with_intercept.T @ X_with_intercept)
    mse = np.sum(residuals**2) / (len(y) - len(coefficients) - 1)  # -1 for intercept
    var_covar = mse * XtX_inv
    
    # Extract standard errors for coefficients (skip intercept)
    se_coefficients = np.sqrt(np.diag(var_covar)[1:])  # Skip intercept
    
    # Calculate confidence intervals
    ci_lower = np.exp(coefficients - 1.96 * se_coefficients)
    ci_upper = np.exp(coefficients + 1.96 * se_coefficients)
    
except (linalg.LinAlgError, ValueError) as e:
    print(f"Warning: Could not calculate confidence intervals due to singular matrix: {e}")
    print("This may happen when filtering to a single center or with insufficient data variation.")
    print("Proceeding with odds ratios only (no confidence intervals).")
    
    # Set confidence intervals to NaN
    ci_lower = np.full_like(coefficients, np.nan)
    ci_upper = np.full_like(coefficients, np.nan)

# Create complete odds ratio table for reference
complete_odds_ratio_df = pl.DataFrame({
    'Variable': predictor_cols,
    'Coefficient': coefficients,
    'Odds_Ratio': odds_ratios,
    'OR_CI_Lower': ci_lower,
    'OR_CI_Upper': ci_upper,
})

# Create AFI-focused table (main result of interest)
afi_odds_ratio_df = pl.DataFrame({
    'Variable': [predictor_cols[0]],  # AFI is first predictor
    'Coefficient': [coefficients[0]],
    'Odds_Ratio': [odds_ratios[0]],
    'OR_CI_Lower': [ci_lower[0]],
    'OR_CI_Upper': [ci_upper[0]],
})

print("\n" + "="*60)
print("LOGISTIC REGRESSION RESULTS")
print("="*60)
print(f"Model: AFI + Centro Studi (control variable) → Outcome 1")
print(f"Note: Centro Studi included to control for confounding factors")
print("\nPrimary Result - AFI Odds Ratio (95% CI):")
print(afi_odds_ratio_df)

print(f"\nAFI Odds Ratio: {odds_ratios[0]:.3f}")
if not np.isnan(ci_lower[0]):
    print(f"95% CI: {ci_lower[0]:.3f}-{ci_upper[0]:.3f}")
    if odds_ratios[0] > 1:
        print(f"→ Higher AFI values are associated with {odds_ratios[0]:.3f}x higher odds of Outcome 1")
    else:
        print(f"→ Higher AFI values are associated with {(1/odds_ratios[0]):.3f}x lower odds of Outcome 1")
else:
    print("95% CI: Could not be calculated (insufficient data variation)")
    print(f"→ Odds ratio: {odds_ratios[0]:.3f} (interpret with caution)")

# Complete results for reference
print(f"\nComplete Model Results (for reference):")
print(complete_odds_ratio_df)

# %% ROC Curve Analysis for Logistic Regression
%matplotlib inline

# Get predicted probabilities
y_pred_proba = log_reg.predict_proba(X_scaled)[:, 1]
y_pred = log_reg.predict(X_scaled)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
auc_score = roc_auc_score(y, y_pred_proba)
u_stat, p_two_sided = mannwhitneyu(y, y_pred_proba, alternative='two-sided')

# Create ROC curve plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
# ROC Curve
ax1.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f}, p={p_two_sided:.4g})')
ax1.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', alpha=0.7)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate (1 - Specificity)')
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_title('ROC Curve: AFI (adjusted for Centro Studi)')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Odds Ratio Forest Plot - Only AFI (main result)
if not np.isnan(ci_lower[0]):
    ax2.errorbar([odds_ratios[0]], [0], 
                xerr=[[odds_ratios[0] - ci_lower[0]], [ci_upper[0] - odds_ratios[0]]],
                fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8, color='blue')
    ci_text = f'OR = {odds_ratios[0]:.3f}\n(95% CI: {ci_lower[0]:.3f}-{ci_upper[0]:.3f})'
else:
    ax2.scatter([odds_ratios[0]], [0], s=100, color='blue', marker='o')
    ci_text = f'OR = {odds_ratios[0]:.3f}\n(CI: Not available)'

ax2.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='No effect (OR=1)')
ax2.set_yticks([0])
ax2.set_yticklabels(['AFI\n(adjusted for Centro Studi)'])
ax2.set_xlabel('Odds Ratio')
ax2.set_title('AFI Odds Ratio with 95% Confidence Interval')
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')
ax2.legend()

# Add text with OR value
ax2.text(odds_ratios[0], -0.1, ci_text, 
         ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))


# %% Performance Metrics Table

# Calculate additional metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Find optimal threshold using Youden's index
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
optimal_sensitivity = tpr[optimal_idx]
optimal_specificity = 1 - fpr[optimal_idx]

# Calculate metrics at optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

performance_metrics = pl.DataFrame({
    'Metric': [
        'AUC',
        'Accuracy (optimal threshold)',
        'Sensitivity (optimal threshold)',
        'Specificity (optimal threshold)',
        'Precision (optimal threshold)', 
        'F1-Score (optimal threshold)',
        'Optimal Threshold'
    ],
    'Value': [
        auc_score,
        accuracy_score(y, y_pred_optimal),
        recall_score(y, y_pred_optimal),
        1 - fpr[optimal_idx],
        precision_score(y, y_pred_optimal),
        f1_score(y, y_pred_optimal),
        optimal_threshold
    ]
})

print("\n" + "="*60)
print("ROC CURVE & PERFORMANCE METRICS")
print("="*60)
print(f"\nAUC Score: {auc_score:.3f}")
print(f"Optimal Threshold (Youden's Index): {optimal_threshold:.3f}")
print(f"Sensitivity at optimal threshold: {optimal_sensitivity:.3f}")
print(f"Specificity at optimal threshold: {optimal_specificity:.3f}")

print("\nDetailed Performance Metrics:")
print(performance_metrics)

# Confusion Matrix at optimal threshold
print(f"\nConfusion Matrix (threshold = {optimal_threshold:.3f}):")
cm = confusion_matrix(y, y_pred_optimal)
print(cm)

print(f"\nClassification Report (threshold = {optimal_threshold:.3f}):")
print(classification_report(y, y_pred_optimal))

# %% ROC Curve Data Table
roc_table = pl.DataFrame({
    'Threshold': thresholds,
    'False_Positive_Rate': fpr,
    'True_Positive_Rate': tpr,
    'Specificity': 1 - fpr,
    'Sensitivity': tpr
})

print("\n" + "="*60)
print("ROC CURVE DATA TABLE (first 10 rows)")
print("="*60)
print(roc_table.head(10))

# Export results to clipboard for easy copying
print("\n" + "="*60)
print("SUMMARY FOR EXPORT")
print("="*60)
summary_results = pl.DataFrame({
    'Analysis': ['Logistic Regression: AFI (adjusted for Centro Studi) vs Outcome 1'],
    'AUC': [auc_score],
    'AFI_OR': [odds_ratios[0]],
    'AFI_CI_Lower': [ci_lower[0]],
    'AFI_CI_Upper': [ci_upper[0]],
    'Optimal_Threshold': [optimal_threshold],
    'Sensitivity': [optimal_sensitivity],
    'Specificity': [optimal_specificity]
})

print(summary_results)
if not np.isnan(ci_lower[0]):
    print(f"\nKey Finding: AFI OR = {odds_ratios[0]:.3f} (95% CI: {ci_lower[0]:.3f}-{ci_upper[0]:.3f})")
else:
    print(f"\nKey Finding: AFI OR = {odds_ratios[0]:.3f} (CI: Not available)")
print(f"AUC = {auc_score:.3f}")
print("Note: Results are adjusted for Centro Studi as a control variable")






# %% 
