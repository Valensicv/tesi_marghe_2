import polars as pl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, classification_report
)
from scipy import stats
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import List, Dict, Tuple, Optional, Union
import warnings


def prepare_logistic_regression_data(
    data: pl.DataFrame,
    predictor_cols: List[str],
    control_cols: List[str],
    target_col: str,
    print_info: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Prepara i dati per la regressione logistica.
    
    Args:
        data: DataFrame Polars con i dati
        predictor_cols: Lista delle colonne predittrici principali
        control_cols: Lista delle colonne di controllo
        target_col: Nome della colonna target
        print_info: Se stampare informazioni sui dati
    
    Returns:
        Tuple con: X (features), y (target), encoding_info (dizionario con info encoding)
    """
    # Combina predittori e controlli
    all_vars = predictor_cols + control_cols + [target_col]
    
    # Prepara i dati
    regression_data = (
        data
        .select(all_vars)
        .drop_nulls()
    )
    
    if print_info:
        print(f"Shape after dropping nulls: {regression_data.shape}")
        print(f"Target distribution:")
        print(regression_data.group_by(target_col).agg(pl.len().alias('count')))
        
        for col in control_cols:
            print(f"\n{col} distribution:")
            print(regression_data.group_by(col).agg(pl.len().alias('count')))
    
    # Converti in pandas
    regression_df = regression_data.to_pandas()
    
    # Encoding delle variabili categoriche
    encoding_info = {}
    encoded_cols = []
    
    for col in predictor_cols + control_cols:
        if regression_df[col].dtype == 'object':
            le = LabelEncoder()
            encoded_col = f"{col}_encoded"
            regression_df[encoded_col] = le.fit_transform(regression_df[col])
            encoded_cols.append(encoded_col)
            encoding_info[col] = {
                "encoder": le,
                "mapping": dict(zip(le.classes_, le.transform(le.classes_))),
            }
            if print_info:
                print(f"\n{col} encoding: {encoding_info[col]['mapping']}")
        else:
            encoded_cols.append(col)
    
    # Prepara features e target
    X = regression_df[encoded_cols]
    y = regression_df[target_col]
    
    return X, y, encoding_info


def fit_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    max_iter: int = 1000
) -> Tuple[LogisticRegression, StandardScaler, np.ndarray]:
    """
    Adatta il modello di regressione logistica.
    
    Args:
        X: Features
        y: Target
        random_state: Seed per riproducibilità
        max_iter: Numero massimo di iterazioni
    
    Returns:
        Tuple con: modello, scaler, coefficienti standardizzati
    """
    # Standardizza le features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Adatta il modello
    log_reg = LogisticRegression(random_state=random_state, max_iter=max_iter)
    log_reg.fit(X_scaled, y)
    
    return log_reg, scaler, X_scaled


def calculate_odds_ratios(
    log_reg: LogisticRegression,
    X_scaled: np.ndarray,
    y: pd.Series,
    feature_names: List[str],
    confidence_level: float = 0.95
) -> pl.DataFrame:
    """
    Calcola gli odds ratios con intervalli di confidenza.
    
    Args:
        log_reg: Modello di regressione logistica addestrato
        X_scaled: Features standardizzate
        y: Target
        feature_names: Nomi delle features
        confidence_level: Livello di confidenza (default 0.95)
    
    Returns:
        DataFrame con odds ratios e intervalli di confidenza
    """
    coefficients = log_reg.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    # Calcola intervalli di confidenza
    try:
        # Aggiungi colonna intercept
        X_with_intercept = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        
        # Calcola probabilità predette
        y_pred_proba = log_reg.predict_proba(X_scaled)[:, 1]
        
        # Calcola residui
        residuals = y - y_pred_proba
        
        # Calcola matrice varianza-covarianza
        XtX_inv = linalg.inv(X_with_intercept.T @ X_with_intercept)
        mse = np.sum(residuals**2) / (len(y) - len(coefficients) - 1)
        var_covar = mse * XtX_inv
        
        # Estrai errori standard per i coefficienti (salta intercept)
        se_coefficients = np.sqrt(np.diag(var_covar)[1:])
        
        # Calcola intervalli di confidenza
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = np.exp(coefficients - z_score * se_coefficients)
        ci_upper = np.exp(coefficients + z_score * se_coefficients)
        
    except (linalg.LinAlgError, ValueError) as e:
        warnings.warn(f"Could not calculate confidence intervals: {e}")
        ci_lower = np.full_like(coefficients, np.nan)
        ci_upper = np.full_like(coefficients, np.nan)
    
    # Crea DataFrame risultato
    result_df = pl.DataFrame({
        'Variable': feature_names,
        'Coefficient': coefficients,
        'Odds_Ratio': odds_ratios,
        'OR_CI_Lower': ci_lower,
        'OR_CI_Upper': ci_upper,
    })
    
    return result_df


def calculate_roc_metrics(
    log_reg: LogisticRegression,
    X_scaled: np.ndarray,
    y: pd.Series
) -> Dict:
    """
    Calcola le metriche ROC e trova il punto ottimo.
    
    Args:
        log_reg: Modello di regressione logistica addestrato
        X_scaled: Features standardizzate
        y: Target
    
    Returns:
        Dizionario con tutte le metriche ROC
    """
    # Calcola probabilità predette
    y_pred_proba = log_reg.predict_proba(X_scaled)[:, 1]
    
    # Calcola curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    auc_score = roc_auc_score(y, y_pred_proba)

    scores_pos = y_pred_proba[y == 1]
    scores_neg = y_pred_proba[y == 0]

    u_stat, p_two_sided = stats.mannwhitneyu(scores_pos, scores_neg, alternative="two-sided")

    # Trova punto ottimo usando indice di Youden
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]

    # Trova punto ottimo usando distanza minima dall'angolo (0,1)
    distances = np.sqrt((1 - tpr) ** 2 + fpr**2)
    optimal_idx_dist = np.argmin(distances)
    optimal_threshold_dist = thresholds[optimal_idx_dist]
    optimal_sensitivity_dist = tpr[optimal_idx_dist]
    optimal_specificity_dist = 1 - fpr[optimal_idx_dist]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc_score": auc_score,
        "u_stat": u_stat,
        "p_value": p_two_sided,  # Questo p-value corrisponde a H0: AUC = 0.5
        "optimal_threshold_youden": optimal_threshold,
        "optimal_sensitivity_youden": optimal_sensitivity,
        "optimal_specificity_youden": optimal_specificity,
        "optimal_threshold_dist": optimal_threshold_dist,
        "optimal_sensitivity_dist": optimal_sensitivity_dist,
        "optimal_specificity_dist": optimal_specificity_dist,
        "y_pred_proba": y_pred_proba,
    }


def calculate_performance_metrics(
    y: pd.Series, y_pred_proba: np.ndarray, threshold: float
) -> pl.DataFrame:
    """
    Calcola le metriche di performance a una soglia specifica.

    Args:
        y: Target reale
        y_pred_proba: Probabilità predette
        threshold: Soglia di classificazione

    Returns:
        DataFrame con le metriche di performance
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Sensitivity": recall_score(y, y_pred),
        "Specificity": 1 - (np.sum((y_pred == 1) & (y == 0)) / np.sum(y == 0)),
        "Precision": precision_score(y, y_pred),
        "F1-Score": f1_score(y, y_pred),
    }

    return pl.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})


def plot_roc_analysis(
    roc_metrics: Dict,
    odds_ratios: pl.DataFrame,
    title: str = "ROC Analysis",
    figsize: Tuple[int, int] = (15, 6),
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Crea il plot ROC e forest plot degli odds ratios.

    Args:
        roc_metrics: Dizionario con metriche ROC
        odds_ratios: DataFrame con odds ratios
        title: Titolo del plot
        figsize: Dimensioni della figura

    Returns:
        Tuple con figura e assi
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ROC Curve
    ax1.plot(
        roc_metrics["fpr"],
        roc_metrics["tpr"],
        color="blue",
        lw=2,
        label=f"ROC Curve (AUC = {roc_metrics['auc_score']:.3f}, p={roc_metrics['p_value']:.4g})",
    )
    ax1.plot([0, 1], [0, 1], color="red", lw=1, linestyle="--", alpha=0.7)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title(f"ROC Curve: {title}")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Forest Plot - Solo primo predittore (risultato principale)
    # Correzione: accedi correttamente ai dati del DataFrame Polars
    main_predictor_row = odds_ratios.row(0, named=True)  # Prendi la prima riga come dizionario
    or_value = main_predictor_row["Odds_Ratio"]
    ci_lower = main_predictor_row["OR_CI_Lower"]
    ci_upper = main_predictor_row["OR_CI_Upper"]
    variable_name = main_predictor_row["Variable"]

    if not np.isnan(ci_lower):
        ax2.errorbar(
            [or_value],
            [0],
            xerr=[[or_value - ci_lower], [ci_upper - or_value]],
            fmt="o",
            capsize=5,
            capthick=2,
            elinewidth=2,
            markersize=8,
            color="blue",
        )
        ci_text = f"OR = {or_value:.3f}\n(95% CI: {ci_lower:.3f}-{ci_upper:.3f})"
    else:
        ax2.scatter([or_value], [0], s=100, color="blue", marker="o")
        ci_text = f"OR = {or_value:.3f}\n(CI: Not available)"

    ax2.axvline(x=1, color="red", linestyle="--", alpha=0.7, label="No effect (OR=1)")
    ax2.set_yticks([0])
    ax2.set_yticklabels([f"{variable_name}\n(adjusted for controls)"])
    ax2.set_xlabel("Odds Ratio")
    ax2.set_title(f"{variable_name} Odds Ratio with 95% CI")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.legend()

    # Aggiungi testo con valore OR
    ax2.text(
        or_value,
        -0.1,
        ci_text,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.tight_layout()
    return fig, (ax1, ax2)


def run_complete_logistic_analysis(
    data: pl.DataFrame,
    predictor_cols: List[str],
    control_cols: List[str],
    target_col: str,
    analysis_name: str = "Logistic Regression Analysis",
    print_results: bool = True,
    plot_results: bool = True,
    random_state: int = 42,
) -> Dict:
    """
    Esegue l'analisi completa di regressione logistica (univariata con controlli).

    Args:
        data: DataFrame Polars con i dati
        predictor_cols: Lista delle colonne predittrici principali
        control_cols: Lista delle colonne di controllo
        target_col: Nome della colonna target
        analysis_name: Nome dell'analisi per i plot
        print_results: Se stampare i risultati
        plot_results: Se creare i plot
        random_state: Seed per riproducibilità

    Returns:
        Dizionario con tutti i risultati dell'analisi
    """
    # Prepara i dati
    X, y, encoding_info = prepare_logistic_regression_data(
        data, predictor_cols, control_cols, target_col, print_info=print_results
    )
    
    # Adatta il modello
    log_reg, scaler, X_scaled = fit_logistic_regression(X, y, random_state)
    
    # Calcola odds ratios
    odds_ratios = calculate_odds_ratios(log_reg, X_scaled, y, X.columns.tolist())
    
    # Calcola metriche ROC
    roc_metrics = calculate_roc_metrics(log_reg, X_scaled, y)
    
    # Calcola metriche di performance
    performance_metrics = calculate_performance_metrics(
        y, roc_metrics['y_pred_proba'], roc_metrics['optimal_threshold_youden']
    )
    
    # Crea plot se richiesto
    if plot_results:
        fig, axes = plot_roc_analysis(roc_metrics, odds_ratios, analysis_name)
        plt.show()
    
    # Stampa risultati se richiesto
    if print_results:
        print("\n" + "="*60)
        print(f"{analysis_name.upper()}")
        print("="*60)
        print(f"Model: {' + '.join(predictor_cols)} + {' + '.join(control_cols)} → {target_col}")
        
        print(f"\nPrimary Result - {predictor_cols[0]} Odds Ratio (95% CI):")
        main_result = odds_ratios.row(0, named=True)
        print(f"OR: {main_result['Odds_Ratio']:.3f}")
        if not np.isnan(main_result['OR_CI_Lower']):
            print(f"95% CI: {main_result['OR_CI_Lower']:.3f}-{main_result['OR_CI_Upper']:.3f}")
        else:
            print("95% CI: Could not be calculated")
        
        print(f"\nAUC Score: {roc_metrics['auc_score']:.3f}")
        print(f"Mann-Whitney p-value (H0: AUC = 0.5): {roc_metrics['p_value']:.4g}")
        print(f"Optimal Threshold (Youden): {roc_metrics['optimal_threshold_youden']:.3f}")
        print(f"Sensitivity: {roc_metrics['optimal_sensitivity_youden']:.3f}")
        print(f"Specificity: {roc_metrics['optimal_specificity_youden']:.3f}")
        
        print(f"\nComplete Results:")
        print(odds_ratios)
        print(f"\nPerformance Metrics:")
        print(performance_metrics)
    
    # Prepara risultati per export
    summary_results = pl.DataFrame(
        {
            "Analysis": [analysis_name],
            "AUC": [roc_metrics["auc_score"]],
            "AUC_p_value": [roc_metrics["p_value"]],  # Aggiunto p-value
            "Main_Predictor_OR": [odds_ratios.row(0, named=True)["Odds_Ratio"]],
            "Main_Predictor_CI_Lower": [odds_ratios.row(0, named=True)["OR_CI_Lower"]],
            "Main_Predictor_CI_Upper": [odds_ratios.row(0, named=True)["OR_CI_Upper"]],
            "Optimal_Threshold": [roc_metrics["optimal_threshold_youden"]],
            "Sensitivity": [roc_metrics["optimal_sensitivity_youden"]],
            "Specificity": [roc_metrics["optimal_specificity_youden"]],
        }
    )
    
    return {
        'model': log_reg,
        'scaler': scaler,
        'odds_ratios': odds_ratios,
        'roc_metrics': roc_metrics,
        'performance_metrics': performance_metrics,
        'summary_results': summary_results,
        'encoding_info': encoding_info,
        'X_scaled': X_scaled,
        'y': y
    }


def plot_multivariate_forest_plot(
    odds_ratios: pl.DataFrame,
    title: str = "Multivariate Logistic Regression Results",
    figsize: Tuple[int, int] = (10, 8),
) -> Tuple[Figure, Axes]:
    """
    Crea un forest plot per regressione logistica multivariata.

    Args:
        odds_ratios: DataFrame con odds ratios
        title: Titolo del plot
        figsize: Dimensioni della figura

    Returns:
        Tuple con figura e asse
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepara i dati per il plot
    y_positions = np.arange(len(odds_ratios))
    or_values = odds_ratios["Odds_Ratio"].to_numpy()
    ci_lower = odds_ratios["OR_CI_Lower"].to_numpy()
    ci_upper = odds_ratios["OR_CI_Upper"].to_numpy()
    variables = odds_ratios["Variable"].to_numpy()

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
            # Aggiungi testo con OR e CI
            ax.text(
                or_val,
                i + 0.1,
                f"OR = {or_val:.3f}\n(95% CI: {ci_l:.3f}-{ci_u:.3f})",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7),
            )
        else:
            ax.scatter(or_val, i, s=100, color="blue", marker="o")
            ax.text(
                or_val,
                i + 0.1,
                f"OR = {or_val:.3f}\n(CI: Not available)",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7),
            )

    # Linea di riferimento per OR = 1
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.7, label="No effect (OR=1)")

    # Configurazione assi
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables)
    ax.set_xlabel("Odds Ratio (adjusted)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    ax.set_xscale("log")
    ax.legend()

    plt.tight_layout()
    return fig, ax


def run_multivariate_logistic_analysis(
    data: pl.DataFrame,
    predictor_cols: List[str],
    control_cols: List[str],
    target_col: str,
    analysis_name: str = "Multivariate Logistic Regression Analysis",
    print_results: bool = True,
    plot_results: bool = True,
    random_state: int = 42,
) -> Dict:
    """
    Esegue l'analisi di regressione logistica multivariata.

    Args:
        data: DataFrame Polars con i dati
        predictor_cols: Lista delle colonne predittrici (tutte incluse nel modello)
        control_cols: Lista delle colonne di controllo (opzionale, per compatibilità)
        target_col: Nome della colonna target
        analysis_name: Nome dell'analisi per i plot
        print_results: Se stampare i risultati
        plot_results: Se creare i plot
        random_state: Seed per riproducibilità

    Returns:
        Dizionario con tutti i risultati dell'analisi
    """
    # Combina predittori e controlli (in multivariate sono tutti predittori)
    all_predictors = predictor_cols + control_cols

    # Prepara i dati
    X, y, encoding_info = prepare_logistic_regression_data(
        data, all_predictors, [], target_col, print_info=print_results
    )

    # Adatta il modello
    log_reg, scaler, X_scaled = fit_logistic_regression(X, y, random_state)

    # Calcola odds ratios
    odds_ratios = calculate_odds_ratios(log_reg, X_scaled, y, X.columns.tolist())

    # Calcola metriche ROC
    roc_metrics = calculate_roc_metrics(log_reg, X_scaled, y)

    # Calcola metriche di performance
    performance_metrics = calculate_performance_metrics(
        y, roc_metrics["y_pred_proba"], roc_metrics["optimal_threshold_youden"]
    )

    # Crea plot se richiesto
    if plot_results:
        # Plot ROC
        fig_roc, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ROC Curve
        ax1.plot(
            roc_metrics["fpr"],
            roc_metrics["tpr"],
            color="blue",
            lw=2,
            label=f"ROC Curve (AUC = {roc_metrics['auc_score']:.3f}, p={roc_metrics['p_value']:.4g})",
        )
        ax1.plot([0, 1], [0, 1], color="red", lw=1, linestyle="--", alpha=0.7)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate (1 - Specificity)")
        ax1.set_ylabel("True Positive Rate (Sensitivity)")
        ax1.set_title(f"ROC Curve: {analysis_name}")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Forest plot multivariato
        plot_multivariate_forest_plot(odds_ratios, analysis_name, figsize=(10, 6))
        ax2 = plt.gca()

        plt.show()

    # Stampa risultati se richiesto
    if print_results:
        print("\n" + "=" * 60)
        print(f"{analysis_name.upper()}")
        print("=" * 60)
        print(f"Model: {' + '.join(all_predictors)} → {target_col}")
        print(f"Note: All odds ratios are adjusted for other variables in the model")

        print(f"\nAdjusted Odds Ratios (95% CI):")
        for row in odds_ratios.iter_rows(named=True):
            print(f"{row['Variable']}: OR = {row['Odds_Ratio']:.3f}", end="")
            if not np.isnan(row["OR_CI_Lower"]):
                print(f" (95% CI: {row['OR_CI_Lower']:.3f}-{row['OR_CI_Upper']:.3f})")
            else:
                print(" (CI: Not available)")

        print(f"\nModel Performance:")
        print(f"AUC Score: {roc_metrics['auc_score']:.3f}")
        print(f"Mann-Whitney p-value (H0: AUC = 0.5): {roc_metrics['p_value']:.4g}")
        print(f"Optimal Threshold (Youden): {roc_metrics['optimal_threshold_youden']:.3f}")
        print(f"Sensitivity: {roc_metrics['optimal_sensitivity_youden']:.3f}")
        print(f"Specificity: {roc_metrics['optimal_specificity_youden']:.3f}")

        print(f"\nComplete Results:")
        print(odds_ratios)
        print(f"\nPerformance Metrics:")
        print(performance_metrics)

    # Prepara risultati per export
    summary_results = pl.DataFrame(
        {
            "Analysis": [analysis_name],
            "AUC": [roc_metrics["auc_score"]],
            "AUC_p_value": [roc_metrics["p_value"]],
            "Optimal_Threshold": [roc_metrics["optimal_threshold_youden"]],
            "Sensitivity": [roc_metrics["optimal_sensitivity_youden"]],
            "Specificity": [roc_metrics["optimal_specificity_youden"]],
        }
    )

    return {
        "model": log_reg,
        "scaler": scaler,
        "odds_ratios": odds_ratios,
        "roc_metrics": roc_metrics,
        "performance_metrics": performance_metrics,
        "summary_results": summary_results,
        "encoding_info": encoding_info,
        "X_scaled": X_scaled,
        "y": y,
    }


def create_roc_data_table(roc_metrics: Dict) -> pl.DataFrame:
    """
    Crea una tabella con tutti i dati della curva ROC.
    
    Args:
        roc_metrics: Dizionario con metriche ROC
    
    Returns:
        DataFrame con dati ROC
    """
    return pl.DataFrame({
        'Threshold': roc_metrics['thresholds'],
        'False_Positive_Rate': roc_metrics['fpr'],
        'True_Positive_Rate': roc_metrics['tpr'],
        'Specificity': 1 - roc_metrics['fpr'],
        'Sensitivity': roc_metrics['tpr']
    }) 