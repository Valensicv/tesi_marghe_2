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

# %% import log_reg_utilities

from tesi_marghe.utils.log_reg import run_complete_logistic_analysis, run_multivariate_logistic_analysis

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


# # Metriche ROC
# print(f"AUC: {results_afi['roc_metrics']['auc_score']:.3f}")

# # Risultati di sintesi per export
# print(results_afi['summary_results'])

# # Tabella dati ROC
# from tesi_marghe.utils.log_reg import create_roc_data_table
# roc_table = create_roc_data_table(results_afi['roc_metrics'])
# print(roc_table.head(10))

# %% svr - 


results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['svr'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="SVR (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# %% pasc outcome 1


results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['pasc'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="pasc (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# %% padc

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['padc'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="padc (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# %% co


results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['co'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="CO (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# %% ci

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['ci'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="Ci (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# %% sv

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['sv'],
    control_cols=['centro_studi'],
    target_col='outcome_1',
    analysis_name="sv (adjusted for Centro Studi) vs Outcome 1",
    print_results=False
)

# %% 

# %% Logistic Regression Analysis: AFI & Centro Studi vs Outcome 2

# Analisi con AFI come predittore principale e Centro Studi come controllo
results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['afi'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="AFI (adjusted for Centro Studi) vs Outcome 2",
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

# %% svr - 


results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['svr'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="SVR (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)

# %% pasc outcome 2


results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['pasc'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="pasc (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)

# %% padc outcome 2

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['padc'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="padc (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)

# %% co
results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['co'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="CO (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)

# %% ci

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['ci'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="Ci (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)

# %% sv

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['sv'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="sv (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)

# %%  cpr

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['cpr'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="cpr (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)

# %% qvo - outcome 2

results_afi = run_complete_logistic_analysis(
    data=data,
    predictor_cols=['qvo'],
    control_cols=['centro_studi'],
    target_col='outcome_2',
    analysis_name="qvo (adjusted for Centro Studi) vs Outcome 2",
    print_results=False
)
