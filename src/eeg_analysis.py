# =============================================================================
# eeg_analysis.py
# EEG Signal Processing & Visualization Utilities
# Universidad / Tesis de Grado
# =============================================================================
# USAGE (in Jupyter Notebook):
#   from eeg_analysis import (
#       plot_demographics, plot_sessions, plot_integrity_1020,
#       plot_initial_audit, plot_exclusion,
#       get_initial_audit, estimate_average_ictal_durations,
#       create_sample, get_top_features_ranking,
#       plot_spearman_heatmap, plot_pca_rank, plot_umap, plot_umap_3d,
#       run_kruskal, plot_topomap, plot_kde,
#       plot_seizure, get_train_audit, plot_train_comparison
#   )
# =============================================================================

# --- Standard Library ---
import io
import base64
from pathlib import Path
from collections import defaultdict

# --- Scientific Computing ---
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import Rbf
from scipy.stats import spearmanr

# --- Machine Learning & Feature Selection ---
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.multitest import multipletests
from mrmr import mrmr_classif

# --- Visualization ---
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

# --- Notebook Display ---
from IPython.display import HTML, display

# --- EEG / Signal ---
import mne
mne.set_log_level('WARNING')

# --- UMAP & Plotly (3-D) ---
import umap
import plotly.graph_objects as go
import plotly.express as px

# --- Parquet / Arrow ---
import pyarrow.parquet as pq
import glob
import os


# =============================================================================
# SECTION 1 — METADATA DEMOGRAPHICS
# =============================================================================

def plot_demographics(path_metadata):
    """
    Two-panel figure: age histogram (left) and gender bar chart (right).

    Parameters
    ----------
    path_metadata : str
        Path to the Parquet file containing patient metadata
        (must include 'age' and 'gender' columns).
    """
    # 1. Load & clean
    df_p = pd.read_parquet(path_metadata)
    df_clean = df_p[df_p['age'] < 99].copy()
    df_clean['gender_label'] = df_clean['gender'].map({1: 'Masculino', 2: 'Femenino'})

    med_age = df_clean['age'].median()
    iqr_val = df_clean['age'].quantile(0.75) - df_clean['age'].quantile(0.25)

    # 2. Aesthetics
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel — Age histogram
    counts_age, bins_age, _ = axes[0].hist(
        df_clean['age'], bins=15, color='#1f77b4', edgecolor='black', alpha=0.8
    )
    axes[0].set_ylim(0, 100)

    for count, bin_edge in zip(counts_age, bins_age):
        if count > 0:
            axes[0].text(
                bin_edge + (bins_age[1] - bins_age[0]) / 2, count + 2,
                f'{int(count)}', ha='center', va='bottom', fontsize=9, fontweight='bold'
            )

    stats_text = f"Mediana: {med_age:.1f}\nIQR: {iqr_val:.1f}"
    axes[0].text(
        0.05, 0.95, stats_text, transform=axes[0].transAxes,
        fontsize=9, fontweight='bold', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.8)
    )

    axes[0].set_title("Distribución de Edad", fontsize=11, fontweight='bold', pad=12)
    axes[0].set_xlabel("Edad (Años)", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Número de Pacientes", fontsize=10, fontweight='bold')
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)

    # Right panel — Gender bar chart
    g_counts = df_clean['gender_label'].value_counts()
    g_pct = (g_counts / g_counts.sum()) * 100
    colors_g = ['#1f77b4', '#d62728']

    bars_gen = axes[1].bar(
        g_pct.index, g_pct.values, color=colors_g, edgecolor='black', alpha=0.8, width=0.6
    )

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylim(0, 100)
    axes[1].set_yticks(np.arange(0, 101, 10))
    axes[1].set_title("Distribución por Género (%)", fontsize=11, fontweight='bold', pad=12)
    axes[1].set_ylabel("Porcentaje (%)", fontsize=10, fontweight='bold', labelpad=12)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)

    for i, bar in enumerate(bars_gen):
        height = bar.get_height()
        abs_count = g_counts.iloc[i]
        axes[1].text(
            bar.get_x() + bar.get_width() / 2., height + 3,
            f'{int(abs_count)}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='black'
        )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig)


# =============================================================================
# SECTION 2 — SESSION QUALITY METRICS
# =============================================================================

def plot_sessions(path_metadata):
    """
    Two-panel figure: channel count bar chart (left) and
    sampling-frequency bar chart as % (right).

    Parameters
    ----------
    path_metadata : str
        Path to the Parquet file containing session metadata
        (must include 'n_channels' and 'sfreq' columns).
    """
    df_s = pd.read_parquet(path_metadata)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel — Channel count
    chan_counts = df_s['n_channels'].value_counts().sort_index()
    colors_chan = ['#d62728' if x < 19 else '#1f77b4' for x in chan_counts.index]
    x_labels_chan = [str(int(x)) for x in chan_counts.index]

    bars_chan = axes[0].bar(
        x_labels_chan, chan_counts.values, color=colors_chan, edgecolor='black', alpha=0.8
    )

    try:
        idx_18 = list(chan_counts.index).index(18)
        axes[0].axvline(x=idx_18 + 0.5, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    except ValueError:
        pass

    axes[0].set_title("Integridad del Sistema (Canales)", fontsize=11, fontweight='bold', pad=12)
    axes[0].set_xlabel("Número de Canales", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Número de Sesiones", fontsize=10, fontweight='bold')
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)
    axes[0].tick_params(axis='x', labelsize=8)

    for i, bar in enumerate(bars_chan):
        if chan_counts.index[i] < 19:
            height = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2., height + 10,
                f'{int(height)}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#d62728'
            )

    # Right panel — Sampling frequency (%)
    sfreq_counts = df_s['sfreq'].value_counts().sort_index()
    sfreq_pct = (sfreq_counts / sfreq_counts.sum()) * 100
    x_labels_sfreq = [str(int(x)) for x in sfreq_counts.index]

    bars_sfreq = axes[1].bar(
        x_labels_sfreq, sfreq_pct.values,
        color='#1f77b4', edgecolor='black', alpha=0.8, width=0.6
    )

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Frecuencia de Muestreo (%)", fontsize=11, fontweight='bold', pad=12)
    axes[1].set_xlabel("Frecuencia (Hz)", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Porcentaje de Sesiones (%)", fontsize=10, fontweight='bold', labelpad=12)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)

    for i, bar in enumerate(bars_sfreq):
        height = bar.get_height()
        abs_count = sfreq_counts.iloc[i]
        axes[1].text(
            bar.get_x() + bar.get_width() / 2., height + 2,
            f'{int(abs_count)}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='black'
        )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig)


def plot_integrity_1020(metadata_sessions_path):
    """
    Two-panel figure: 10-20 electrode map (left) and
    valid vs. excluded session counts (right).

    Parameters
    ----------
    metadata_sessions_path : str
        Path to the Parquet file with session metadata
        (must include 'n_channels' column).
    """
    df_s = pd.read_parquet(metadata_sessions_path)
    df_s['is_valid'] = df_s['n_channels'] >= 19

    counts = df_s['is_valid'].map({True: 'Válidas', False: 'Excluidas'}).value_counts()
    counts = counts.reindex(['Válidas', 'Excluidas'])
    total_n = counts.sum()
    pcts = (counts / total_n) * 100

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel — 10-20 scalp diagram
    pos_1020 = {
        'Fp1': (-0.3, 0.9),  'Fp2': (0.3, 0.9),
        'F7':  (-0.8, 0.4),  'F3':  (-0.4, 0.5), 'Fz': (0.0, 0.5), 'F4': (0.4, 0.5), 'F8': (0.8, 0.4),
        'T3':  (-1.0, 0.0),  'C3':  (-0.5, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.5, 0.0), 'T4': (1.0, 0.0),
        'T5':  (-0.8, -0.4), 'P3':  (-0.4, -0.5), 'Pz': (0.0, -0.5), 'P4': (0.4, -0.5), 'T6': (0.8, -0.4),
        'O1':  (-0.3, -0.9), 'O2':  (0.3, -0.9)
    }

    axes[0].add_patch(Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.5))
    axes[0].plot([-0.1, 0, 0.1], [1.0, 1.08, 1.0], color='black', linewidth=1.5)
    axes[0].add_patch(Ellipse((-1.02, 0), 0.12, 0.25, color='black', fill=False))
    axes[0].add_patch(Ellipse((1.02, 0),  0.12, 0.25, color='black', fill=False))

    for label, (px, py) in pos_1020.items():
        axes[0].scatter(px, py, c='white', edgecolor='black', s=190, zorder=11)
        axes[0].text(px, py, label, fontsize=7, ha='center', va='center', fontweight='bold', zorder=12)

    axes[0].set_title("Sistema Internacional 10-20 (19 ch)", fontsize=11, fontweight='bold', pad=12)
    axes[0].set_xlim(-1.2, 1.2)
    axes[0].set_ylim(-1.2, 1.2)
    axes[0].set_box_aspect(1)
    axes[0].axis('off')

    # Right panel — Session validity bar chart
    colors_bars = ['#1f77b4', '#d62728']
    bars = axes[1].bar(counts.index, counts.values, color=colors_bars, edgecolor='black', alpha=0.8, width=0.6)

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylim(0, counts.max() * 1.15)
    axes[1].set_title("Integridad Técnica (N-Sesiones)", fontsize=11, fontweight='bold', pad=12)
    axes[1].set_ylabel("Número de Sesiones", fontsize=10, fontweight='bold', labelpad=12)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)

    for i, bar in enumerate(bars):
        val_pct = pcts.iloc[i]
        height = bar.get_height()
        label_text = f'{val_pct:.1f}%' if val_pct >= 0.1 else '< 0.1%'
        axes[1].text(
            bar.get_x() + bar.get_width() / 2., height + (counts.max() * 0.02),
            label_text, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black'
        )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig)


# =============================================================================
# SECTION 3 — ANNOTATION AUDIT
# =============================================================================

def _merge_intervals(intervals):
    """Merge overlapping time intervals and return total covered duration."""
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    merged = [list(intervals[0])]
    for current in intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(list(current))
    return sum(stop - start for start, stop in merged)


def get_initial_audit(csv_root_path, metadata_path):
    """
    Compute per-label total annotated duration (seconds) across all valid
    patients, merging overlapping channel annotations to avoid double-counting.

    Parameters
    ----------
    csv_root_path : str
        Root directory containing per-session CSV annotation files.
    metadata_path : str
        Path to the Parquet metadata file
        (patients with split_final == -1 are excluded).

    Returns
    -------
    pd.DataFrame
        Index: label name.
        Columns: 'Segundos', 'Prop (%)'.
        Sorted by 'Segundos' descending.
    """
    df_meta = pd.read_parquet(metadata_path)
    valid_patients = set(df_meta[df_meta['split_final'] != -1]['patient_num_id'].astype(str))

    all_csvs = list(Path(csv_root_path).rglob("*.csv"))
    csv_files = [f for f in all_csvs if f.parent.parent.name in valid_patients]

    global_durations = defaultdict(float)

    for f in csv_files:
        file_intervals = defaultdict(list)
        try:
            with open(f, 'r') as file:
                start_idx_found = False
                col_map = {}

                for line in file:
                    if line.startswith('#') or not line.strip():
                        continue

                    parts = [p.strip().lower() for p in line.split(',')]

                    if not start_idx_found:
                        if 'start_time' in parts:
                            col_map = {n: idx for idx, n in enumerate(parts)}
                            start_idx_found = True
                        continue

                    if len(parts) < 4:
                        continue
                    try:
                        lbl   = parts[col_map['label']].lower()
                        start = float(parts[col_map['start_time']])
                        stop  = float(parts[col_map['stop_time']])
                        file_intervals[lbl].append((start, stop))
                    except Exception:
                        continue

            for lbl, intervals in file_intervals.items():
                global_durations[lbl] += _merge_intervals(intervals)

            file_intervals.clear()

        except Exception:
            continue

    if not global_durations:
        return pd.DataFrame()

    df_res = pd.DataFrame([{'Label': k, 'Segundos': v} for k, v in global_durations.items()])
    df_res = df_res.set_index('Label')
    total_total = df_res['Segundos'].sum()
    df_res['Prop (%)'] = (df_res['Segundos'] / total_total * 100).round(2)

    return df_res.sort_values(by='Segundos', ascending=False)


def estimate_average_ictal_durations(csv_root_path, metadata_path):
    """
    Descriptive statistics for individual ictal event durations,
    excluding background ('bckg') and invalidated patients.

    Parameters
    ----------
    csv_root_path : str
        Root directory containing per-session CSV annotation files.
    metadata_path : str
        Path to the Parquet metadata file.

    Returns
    -------
    pd.DataFrame
        Index: label name.
        Columns: N_Eventos, Mediana_Seg, Promedio_Seg, Min_Seg, Max_Seg,
                 Duracion_Total_Seg.
        Sorted by 'Mediana_Seg' descending.
    """
    df_meta = pd.read_parquet(metadata_path)
    valid_patients = set(df_meta[df_meta['split_final'] != -1]['patient_num_id'].astype(str))

    all_csvs = list(Path(csv_root_path).rglob("*.csv"))
    csv_files = [f for f in all_csvs if f.parent.parent.name in valid_patients]

    if not csv_files:
        return pd.DataFrame()

    event_data = defaultdict(list)

    for f in csv_files:
        try:
            with open(f, 'r') as file:
                start_idx_found = False
                col_map = {}
                raw_intervals = []

                for line in file:
                    if line.startswith('#') or not line.strip():
                        continue

                    parts = [p.strip().lower() for p in line.split(',')]

                    if not start_idx_found:
                        if 'start_time' in parts:
                            col_map = {n: idx for idx, n in enumerate(parts)}
                            start_idx_found = True
                        continue

                    if len(parts) < 4:
                        continue
                    try:
                        lbl = parts[col_map['label']].lower()
                        if lbl == 'bckg':
                            continue
                        start = float(parts[col_map['start_time']])
                        stop  = float(parts[col_map['stop_time']])
                        raw_intervals.append((start, stop, lbl))
                    except Exception:
                        continue

                for start, stop, lbl in set(raw_intervals):
                    event_data[lbl].append(stop - start)

        except Exception:
            continue

    if not event_data:
        return pd.DataFrame()

    report = []
    for lbl, durs in event_data.items():
        report.append({
            'Label':             lbl,
            'N_Eventos':         len(durs),
            'Mediana_Seg':       np.median(durs),
            'Promedio_Seg':      np.mean(durs),
            'Min_Seg':           np.min(durs),
            'Max_Seg':           np.max(durs),
            'Duracion_Total_Seg': np.sum(durs)
        })

    df_res = pd.DataFrame(report).set_index('Label').sort_values(by='Mediana_Seg', ascending=False)
    return df_res


def plot_initial_audit(df_sessions_path, df_natural_raw):
    """
    Two-panel audit figure: TCP montage configurations (left) and
    top-5 original event labels by duration (right).

    Parameters
    ----------
    df_sessions_path : str
        Path to the Parquet session metadata file
        (must include 'config_type' column).
    df_natural_raw : pd.DataFrame
        Output of `get_initial_audit()` — must have 'Segundos' and 'Prop (%)'.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel — TCP configurations
    df_s = pd.read_parquet(df_sessions_path)
    config_counts = df_s['config_type'].value_counts().sort_index()
    config_pct = (config_counts / config_counts.sum()) * 100

    colors_conf = ['#34495e', '#7f8c8d', '#95a5a6']
    bars1 = axes[0].bar(
        config_counts.index, config_pct.values,
        color=colors_conf, edgecolor='black', alpha=0.8, width=0.6
    )

    axes[0].set_title("Configuraciones de Montaje (TCP)", fontsize=11, fontweight='bold', pad=15)
    axes[0].set_ylabel("Proporción (%)", fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks(np.arange(0, 101, 10))
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.4)

    for i, bar in enumerate(bars1):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
            f'{int(config_counts.iloc[i])}', ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )

    # Right panel — Top 5 raw event labels
    df_top5 = df_natural_raw.sort_values(by='Segundos', ascending=False).head(5)
    colors_raw = sns.color_palette("viridis", n_colors=5)

    bars2 = axes[1].bar(
        df_top5.index, df_top5['Prop (%)'],
        color=colors_raw, edgecolor='black', alpha=0.8, width=0.6
    )

    axes[1].set_title("Top 5 Eventos (Etiquetas Originales)", fontsize=11, fontweight='bold', pad=15)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel("Proporción del Dataset (%)", fontsize=10, fontweight='bold', labelpad=12)
    axes[1].set_ylim(0, 110)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.4)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        segundos = df_top5['Segundos'].iloc[i]
        axes[1].text(
            bar.get_x() + bar.get_width() / 2., height + 2,
            f'{int(segundos)}s', ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig)


# =============================================================================
# SECTION 4 — RAW EEG EXCLUSION PREVIEW
# =============================================================================

def plot_exclusion(tusz_root_path):
    """
    Side-by-side 10-second EEG traces for two hard-coded exclusion cases.

    Parameters
    ----------
    tusz_root_path : str
        Root path of the TUSZ dataset
        (expects sub-path 01_Raw_Consolidated/PATIENT/SESSION/FILE.edf).
    """
    root = Path(tusz_root_path)

    cases = [
        {"path": root / "01_Raw_Consolidated/50/s001_2003/aaaaabdi_s001_t000.edf",  "name": "Paciente 50"},
        {"path": root / "01_Raw_Consolidated/209/s001_2007/aaaaafwz_s001_t000.edf", "name": "Paciente 209"}
    ]

    target_electrodes = ['FP1', 'C3', 'CZ', 'P3', 'O1']

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    plt.subplots_adjust(wspace=0.15)

    fixed_offset = 800  # µV

    for i, case in enumerate(cases):
        ax = axes[i]
        if not case["path"].exists():
            ax.text(0.5, 0.5, f"No encontrado:\n{case['name']}", ha='center', va='center', fontsize=8)
            continue

        try:
            raw = mne.io.read_raw_edf(case["path"], preload=False, verbose=False)

            picks = []
            for e in target_electrodes:
                m = [ch for ch in raw.ch_names if e in ch.upper()]
                if m:
                    picks.append(m[0])

            raw.load_data().crop(tmin=0, tmax=10).pick(picks)
            data_uv = raw.get_data() * 1e6
            times = raw.times

            for j, ch_name in enumerate(picks):
                ax.plot(times, data_uv[j] + (j * fixed_offset), color='black', linewidth=0.8)

                if i == 0:
                    clean_name = ch_name.replace('EEG ', '').split('-')[0]
                    ax.text(-0.5, (j * fixed_offset), clean_name,
                            fontsize=8, va='center', ha='right', fontweight='bold')

            ax.set_title(case["name"], fontsize=10, fontweight='bold', pad=8)
            ax.set_xlabel("Tiempo (s)", fontsize=8, fontweight='bold')
            ax.set_xlim(0, 10)
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.tick_params(axis='x', labelsize=7)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', fontsize=7)

    axes[0].set_yticks([])

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel("Amplitud (μV)", fontsize=8, fontweight='bold', labelpad=8)

    y_ticks_pos = np.arange(0, len(target_electrodes) * fixed_offset, fixed_offset)
    axes[1].set_yticks(y_ticks_pos)
    axes[1].tick_params(axis='y', labelsize=7)
    axes[1].set_yticklabels(["" for _ in y_ticks_pos], fontsize=7)

    plt.tight_layout()
    _display_figure(fig, dpi=120)


# =============================================================================
# SECTION 5 — DATASET SAMPLING
# =============================================================================

def create_sample(path_04, path_metadata_patients, n_per_macro=1500, seed=42):
    """
    Balanced sample of EEG feature windows from training patients.

    Draws up to `n_per_macro` windows per macro-class (background, focal,
    generalized) using a proportional per-patient quota with a deficit
    compensation pass.

    Parameters
    ----------
    path_04 : str
        Root directory containing Parquet feature files under
        version=v2_augmented_labels/PATIENT_ID/*.parquet.
    path_metadata_patients : str
        Path to the Parquet metadata file
        (training patients have split_final == 0).
    n_per_macro : int
        Target number of windows per macro-class (default 1500).
    seed : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame or None
        Shuffled DataFrame with columns from the feature files plus
        'original_class', 'macro_class', and 'patient_id'.
    """
    mapping = {
        'bckg': 'background',
        'fnsz': 'focal',   'cpsz': 'focal',   'spsz': 'focal',
        'gnsz': 'generalized', 'tcsz': 'generalized', 'tnsz': 'generalized',
        'absz': 'generalized', 'mysz': 'generalized'
    }

    df_meta = pd.read_parquet(path_metadata_patients)
    train_patients = set(df_meta[df_meta['split_final'] == 0]['patient_num_id'].astype(int).unique())

    all_files = glob.glob(os.path.join(path_04, "**", "*.parquet"), recursive=True)
    files_by_patient = {}
    version_folder = "version=v2_augmented_labels"

    for f in all_files:
        parts = Path(f).parts
        if version_folder in parts:
            idx = parts.index(version_folder)
            try:
                patient_id = int(parts[idx + 1])
                if patient_id in train_patients:
                    if patient_id not in files_by_patient:
                        files_by_patient[patient_id] = []
                    files_by_patient[patient_id].append(f)
            except (ValueError, IndexError):
                continue

    active_patients = list(files_by_patient.keys())
    n_patients = len(active_patients)

    if n_patients == 0:
        return None

    base_quota = int(np.ceil(n_per_macro / n_patients))

    np.random.seed(seed)
    np.random.shuffle(active_patients)

    buckets = {'background': [], 'focal': [], 'generalized': []}
    counts_global = {'background': 0, 'focal': 0, 'generalized': 0}

    # First pass — per-patient quota
    for p_id in active_patients:
        patient_files = files_by_patient[p_id]
        np.random.shuffle(patient_files)

        patient_yield = {'background': 0, 'focal': 0, 'generalized': 0}

        for f in patient_files:
            try:
                temp_df = pd.read_parquet(f)
                temp_df['original_class'] = temp_df['label'].str.strip().str.lower()
                temp_df['macro_class']    = temp_df['original_class'].map(mapping)
                temp_df = temp_df.dropna(subset=['macro_class'])

                for m_cls in buckets.keys():
                    if patient_yield[m_cls] < base_quota and counts_global[m_cls] < n_per_macro:
                        needed_patient = base_quota - patient_yield[m_cls]
                        needed_global  = n_per_macro - counts_global[m_cls]
                        take_limit = min(needed_patient, needed_global)

                        available = temp_df[temp_df['macro_class'] == m_cls]
                        if not available.empty:
                            take = min(len(available), take_limit)
                            sampled_df = available.sample(n=take, random_state=seed)
                            sampled_df['patient_id'] = p_id
                            buckets[m_cls].append(sampled_df)
                            counts_global[m_cls] += take
                            patient_yield[m_cls]  += take
                del temp_df
            except Exception:
                continue

    # Second pass — deficit compensation
    if any(counts_global[m_cls] < n_per_macro for m_cls in counts_global):
        for p_id in active_patients:
            if all(counts_global[m_cls] >= n_per_macro for m_cls in counts_global):
                break

            patient_files = files_by_patient[p_id]
            for f in patient_files:
                try:
                    temp_df = pd.read_parquet(f)
                    temp_df['original_class'] = temp_df['label'].str.strip().str.lower()
                    temp_df['macro_class']    = temp_df['original_class'].map(mapping)
                    temp_df = temp_df.dropna(subset=['macro_class'])

                    for m_cls in buckets.keys():
                        if counts_global[m_cls] < n_per_macro:
                            needed_global = n_per_macro - counts_global[m_cls]
                            available = temp_df[temp_df['macro_class'] == m_cls]
                            if not available.empty:
                                take = min(len(available), needed_global)
                                sampled_df = available.sample(n=take, random_state=seed)
                                sampled_df['patient_id'] = p_id
                                buckets[m_cls].append(sampled_df)
                                counts_global[m_cls] += take
                    del temp_df
                except Exception:
                    continue

    res_list = [pd.concat(buckets[m]) for m in buckets if buckets[m]]
    if not res_list:
        return None

    df_final = pd.concat(res_list, ignore_index=True)
    return df_final.sample(frac=1, random_state=seed).reset_index(drop=True)


# =============================================================================
# SECTION 6 — FEATURE SELECTION (mRMR)
# =============================================================================

def get_top_features_ranking(df_sample, target_col='macro_class', top_k=10):
    """
    Rank features using mRMR and compute mutual information + IQR.

    Parameters
    ----------
    df_sample : pd.DataFrame
        Feature matrix including the target column.
    target_col : str
        Name of the target column (default 'macro_class').
    top_k : int
        Number of top features to select (default 10).

    Returns
    -------
    pd.DataFrame
        Index: mRMR Rank (1-based).
        Columns: Feature, Mutual Info, IQR.
    """
    X = df_sample.drop(columns=[target_col]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_sample[target_col]

    selected_features = mrmr_classif(X=X, y=y, K=top_k, show_progress=False)

    X_top = X[selected_features]
    mi_scores = mutual_info_classif(X_top, y, random_state=42)

    ranking_data = []
    for rank, (feature, mi) in enumerate(zip(selected_features, mi_scores), start=1):
        iqr_val = stats.iqr(X_top[feature])
        ranking_data.append({
            'mRMR Rank':   rank,
            'Feature':     feature,
            'Mutual Info': mi,
            'IQR':         iqr_val
        })

    df_ranking = pd.DataFrame(ranking_data).set_index('mRMR Rank')
    print(df_ranking.style.format({'Mutual Info': '{:.4f}', 'IQR': '{:.4f}'}))
    return df_ranking


# =============================================================================
# SECTION 7 — CORRELATION & DIMENSIONALITY REDUCTION VISUALIZATIONS
# =============================================================================

def plot_spearman_heatmap(df_train_sample, top_15_features):
    """
    Lower-triangle Spearman correlation heatmap for selected features.

    Parameters
    ----------
    df_train_sample : pd.DataFrame
        Feature data (must contain all columns in top_15_features).
    top_15_features : list of str
        Ordered list of feature column names.
    """
    df_top = df_train_sample[top_15_features]
    corr_matrix = df_top.corr(method='spearman')

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8
    })
    sns.set_style("white")

    num_vars = len(corr_matrix.columns)
    numeric_labels = list(range(1, num_vars + 1))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='white')

    res = sns.heatmap(
        corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
        vmin=-1, vmax=1, center=0, linewidths=0.5, linecolor='white',
        xticklabels=numeric_labels, yticklabels=numeric_labels,
        cbar_kws={'shrink': .6}, ax=ax
    )

    cbar = res.collections[0].colorbar
    cbar.outline.set_linewidth(0.8)
    cbar.set_label('Coeficiente ρ de Spearman', labelpad=15, weight='bold', fontsize=9, rotation=90)

    ax.tick_params(axis='both', which='major', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig, dpi=120, max_width='65%')


def plot_pca_rank(df_sample, top_features, target_col='macro_class'):
    """
    Two-panel PCA figure using Spearman rank correlation:
    cumulative explained variance (left) and 2-D projection (right).

    Parameters
    ----------
    df_sample : pd.DataFrame
        Feature data including the target column.
    top_features : list of str
        Feature columns to include in the PCA.
    target_col : str
        Target column name (default 'macro_class').
    """
    X = df_sample[top_features]
    y = df_sample[target_col]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    df_ranks = pd.DataFrame(X_scaled).rank()

    corr_spearman, _ = spearmanr(df_ranks)
    eigenvalues, eigenvectors = np.linalg.eig(corr_spearman)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    X_pca   = np.dot(scaler.fit_transform(df_ranks), eigenvectors)
    exp_var = eigenvalues / np.sum(eigenvalues)
    cum_var = np.cumsum(exp_var)
    num_comp = len(cum_var)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor='white')

    # Left panel — Cumulative variance
    x_axis = np.arange(1, num_comp + 1)
    axes[0].plot(x_axis, cum_var * 100, color='#1f77b4', marker='o',
                 markersize=5, linewidth=1.5, markerfacecolor='white', markeredgewidth=1.5)
    axes[0].axhline(y=90, color='#d62728', linestyle='--', alpha=0.7, linewidth=1)

    axes[0].set_title("Varianza Acumulada (Rangos)", fontsize=11, fontweight='bold', pad=12)
    axes[0].set_xlabel("Número de Componentes", fontsize=10, fontweight='bold', labelpad=10)
    axes[0].set_ylabel("Varianza Explicada (%)", fontsize=10, fontweight='bold', labelpad=10)
    axes[0].set_xticks(x_axis)
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks(np.arange(0, 101, 10))
    axes[0].set_yticklabels([f"{int(val)}%" for val in np.arange(0, 101, 10)])
    axes[0].yaxis.grid(True, linestyle='--', alpha=0.3)

    # Right panel — 2-D scatter
    palette = {'background': '#1f77b4', 'focal': '#d62728', 'generalized': '#2ca02c'}

    for label, color in palette.items():
        mask = (y == label).values
        label_es = 'Basal' if label == 'background' else 'Focal' if label == 'focal' else 'Generalizada'
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=color, label=label_es, alpha=0.7, edgecolor='black', s=30, linewidth=0.5)

    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

    pc1_var = exp_var[0] * 100
    pc2_var = exp_var[1] * 100

    axes[1].set_title("Proyección PCA (Spearman)", fontsize=11, fontweight='bold', pad=12)
    axes[1].set_xlabel(f"Componente Principal 1 ({pc1_var:.1f}%)", fontsize=10, fontweight='bold', labelpad=10)
    axes[1].set_ylabel(f"Componente Principal 2 ({pc2_var:.1f}%)", fontsize=10, fontweight='bold', labelpad=10)
    axes[1].yaxis.grid(True, linestyle='--', alpha=0.3)
    axes[1].legend(fontsize=10, loc='best', frameon=False)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig, max_width='85%')


def plot_umap(df_sample, top_features, target_col='macro_class'):
    """
    Side-by-side 2-D UMAP projections: Euclidean (left) and Cosine (right).

    Parameters
    ----------
    df_sample : pd.DataFrame
        Feature data including the target column.
    top_features : list of str
        Feature columns to project.
    target_col : str
        Target column name (default 'macro_class').
    """
    X = df_sample[top_features]
    y = df_sample[target_col].str.lower()
    X_scaled = RobustScaler().fit_transform(X)

    configs = [
        {'metric': 'euclidean', 'nn': 80, 'md': 0.05, 'titulo': 'Distancia Euclidiana'},
        {'metric': 'cosine',    'nn': 50, 'md': 0.10, 'titulo': 'Distancia Coseno'}
    ]

    palette   = {'background': '#1f77b4', 'focal': '#d62728', 'generalized': '#2ca02c'}
    label_map = {'background': 'Basal (Repososo)', 'focal': 'Focal', 'generalized': 'Generalizada'}

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.6
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for i, cfg in enumerate(configs):
        ax = axes[i]

        reducer = umap.UMAP(
            n_neighbors=cfg['nn'], min_dist=cfg['md'],
            metric=cfg['metric'], random_state=42, n_jobs=-1
        )
        embedding = reducer.fit_transform(X_scaled)

        for label_en, color in palette.items():
            mask = (y == label_en).values
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=color, label=label_map[label_en],
                       alpha=0.65, edgecolor='black', s=50, linewidth=0.4)

        ax.set_title(f"UMAP: {cfg['titulo']}", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Dimensión UMAP 1", fontsize=12, fontweight='bold', labelpad=12)
        ax.grid(True, linestyle='--', alpha=0.2)

        param_text = f"# vecinos = {cfg['nn']}\ndistancia mínima = {cfg['md']}"

        if i == 0:
            ax.set_ylabel("Dimensión UMAP 2", fontsize=12, fontweight='bold', labelpad=12)
            ax.text(0.97, 0.97, param_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    fontstyle='italic', alpha=0.8)
        else:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("Dimensión UMAP 2", fontsize=12, fontweight='bold', labelpad=15)
            ax.text(0.03, 0.97, param_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    fontstyle='italic', alpha=0.8)
            ax.legend(fontsize=11, loc='upper right', frameon=False, markerscale=1.2)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig, dpi=250)


def plot_umap_3d(df_sample, top_features, macro_col='macro_class', specific_col='original_class', n_samples=None):
    """
    Interactive Plotly 3-D UMAP projection with macro/specific toggle buttons.

    Parameters
    ----------
    df_sample : pd.DataFrame
        Feature data including macro and specific label columns.
    top_features : list of str
        Feature columns to project.
    macro_col : str
        Column with macro-class labels (default 'macro_class').
    specific_col : str
        Column with specific seizure-type labels (default 'original_class').
    n_samples : int or None
        Subsample size (None = use all rows).
    """
    dicc_nombres_tusz = {
        'bckg': 'Actividad Basal',
        'fnsz': 'Crisis Focal No Específica',
        'gnsz': 'Crisis Generalizada No Específica',
        'cpsz': 'Crisis Parcial Compleja',
        'tcsz': 'Crisis Tónico-Clónica',
        'spsz': 'Crisis Parcial Simple',
        'mysz': 'Crisis Mioclónica',
        'absz': 'Crisis de Ausencia',
        'tnsz': 'Crisis Tónica'
    }

    if n_samples is not None:
        df_work = df_sample.sample(n=min(len(df_sample), n_samples), random_state=42).copy()
    else:
        df_work = df_sample.copy()

    X = df_work[top_features]
    X_scaled = RobustScaler().fit_transform(X)

    label_map = {'background': 'Basal', 'focal': 'Focal', 'generalized': 'Generalizada'}
    df_work['macro_es'] = df_work[macro_col].str.lower().map(lambda x: label_map.get(x, x.capitalize()))
    df_work['specific_hover'] = df_work[specific_col].str.lower().apply(
        lambda x: f"{dicc_nombres_tusz.get(x, x.capitalize())} ({x.upper()})"
    )

    reducer = umap.UMAP(
        n_neighbors=50, min_dist=0.1, n_components=3,
        metric='cosine', random_state=42, n_jobs=-1
    )
    embedding = reducer.fit_transform(X_scaled)
    df_work['u1'], df_work['u2'], df_work['u3'] = embedding[:, 0], embedding[:, 1], embedding[:, 2]

    fig = go.Figure()
    MACRO_PALETTE    = {'Basal': '#1f77b4', 'Focal': '#d62728', 'Generalizada': '#2ca02c'}
    SPECIFIC_TYPES   = sorted(df_work[specific_col].unique())
    SPECIFIC_PALETTE = px.colors.qualitative.Bold

    # Macro traces
    macro_labels = sorted(df_work['macro_es'].unique())
    for label in macro_labels:
        mask = df_work['macro_es'] == label
        fig.add_trace(go.Scatter3d(
            x=df_work.loc[mask, 'u1'], y=df_work.loc[mask, 'u2'], z=df_work.loc[mask, 'u3'],
            mode='markers',
            marker=dict(size=3.5, color=MACRO_PALETTE.get(label, '#7f8c8d'), opacity=0.7,
                        line=dict(width=0.3, color='black')),
            name=label, visible=True, legendgroup=label,
            customdata=np.stack((df_work.loc[mask, 'macro_es'], df_work.loc[mask, 'specific_hover']), axis=-1),
            hovertemplate="<b>Macro:</b> %{customdata[0]}<br><b>Específica:</b> %{customdata[1]}<extra></extra>"
        ))

    # Specific traces
    for i, label in enumerate(SPECIFIC_TYPES):
        mask = df_work[specific_col] == label
        fig.add_trace(go.Scatter3d(
            x=df_work.loc[mask, 'u1'], y=df_work.loc[mask, 'u2'], z=df_work.loc[mask, 'u3'],
            mode='markers',
            marker=dict(size=3.5, color=SPECIFIC_PALETTE[i % len(SPECIFIC_PALETTE)], opacity=0.8,
                        line=dict(width=0.3, color='black')),
            name=label, visible=False, legendgroup=label,
            customdata=np.stack((df_work.loc[mask, 'macro_es'], df_work.loc[mask, 'specific_hover']), axis=-1),
            hovertemplate="<b>Macro:</b> %{customdata[0]}<br><b>Específica:</b> %{customdata[1]}<extra></extra>"
        ))

    n_macro, n_specific = len(macro_labels), len(SPECIFIC_TYPES)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        template="plotly_white",
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.5, xanchor="center", y=0.98, yanchor="top",
            showactive=True, font=dict(size=11),
            buttons=[
                dict(label="Vista Macroclase", method="update",
                     args=[{"visible": [True]  * n_macro + [False] * n_specific}]),
                dict(label="Vista Específica",  method="update",
                     args=[{"visible": [False] * n_macro + [True]  * n_specific}])
            ]
        )],
        scene=dict(
            xaxis_title='UMAP 1', yaxis_title='UMAP 2', zaxis_title='UMAP 3',
            camera=dict(eye=dict(x=0.5, y=-1.8, z=0.15))
        ),
        legend=dict(
            title_text='<b>Clases</b><br> ', font=dict(size=12),
            itemsizing='constant', itemwidth=30, tracegroupgap=2,
            itemclick="toggle", itemdoubleclick="toggleothers",
            yanchor="top", y=0.85, xanchor="left", x=-0.12,
            bgcolor="rgba(255, 255, 255, 0.5)"
        )
    )

    local_path = "umap_3d.html"
    fig.write_html(local_path)

    display(HTML("<script>google.colab.output.setIframeHeight(500, true, {maxHeight: 5000})</script>"))
    display(HTML(f"""
        <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
            <iframe src="{local_path}" width="85%" height="500px" style="border:none;" scrolling="no"></iframe>
        </div>
    """))


# =============================================================================
# SECTION 8 — STATISTICAL TESTS
# =============================================================================

def run_kruskal(df, features, target='macro_class'):
    """
    Kruskal-Wallis H-test with Holm correction and epsilon-squared effect size.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `target` column and all columns in `features`.
    features : list of str
        Feature column names to test.
    target : str
        Target column name (default 'macro_class').

    Returns
    -------
    pd.DataFrame
        One row per feature with columns: Característica, Background, Focal,
        Generalized (Mediana (IQR)), Estadístico H, ε², Magnitud del Efecto.
    """
    results = []
    df[target] = df[target].str.lower()
    classes = ['background', 'focal', 'generalized']

    p_values_raw = []
    for feat in features:
        groups = [df[df[target] == cls][feat].dropna() for cls in classes]
        _, p_val = stats.kruskal(*groups)
        p_values_raw.append(p_val)

    _, p_corrected, _, _ = multipletests(p_values_raw, method='holm')

    for i, feat in enumerate(features):
        row = {'Característica': feat}
        group_data = []

        for cls in classes:
            data = df[df[target] == cls][feat].dropna()
            group_data.append(data)

            med = data.median()
            q1  = data.quantile(0.25)
            q3  = data.quantile(0.75)
            iqr = q3 - q1
            row[cls.capitalize()] = f"{med:.2f} ({iqr:.2f})"

        h_stat, _ = stats.kruskal(*group_data)
        p_adj      = p_corrected[i]

        stars = ""
        if p_adj < 0.001:   stars = "***"
        elif p_adj < 0.01:  stars = "**"
        elif p_adj < 0.05:  stars = "*"

        n           = len(df[feat].dropna())
        epsilon_sq  = h_stat * (n + 1) / (n ** 2 - 1)

        if epsilon_sq < 0.01:   interpret = "Despreciable"
        elif epsilon_sq < 0.08: interpret = "Pequeño"
        elif epsilon_sq < 0.26: interpret = "Moderado"
        else:                   interpret = "Grande"

        row['Estadístico H']       = f"{h_stat:.2f}{stars}"
        row['ε²']                  = f"{epsilon_sq:.2f}"
        row['Magnitud del Efecto'] = interpret
        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# SECTION 9 — TOPOGRAPHIC MAPS & KDE DISTRIBUTIONS
# =============================================================================

def plot_topomap(df, metric_suffix, cbar_label):
    """
    Three-panel scalp topomaps (Basal / Focal / Generalizada) for a given feature.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'macro_class' and feature columns ending with `metric_suffix`.
        Column names should follow the pattern 'E1-E2_<metric_suffix>'.
    metric_suffix : str
        Suffix identifying the feature group (e.g. '_entropy', '_power').
    cbar_label : str
        Label for the shared colour bar.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })

    cols_of_interest = [c for c in df.columns if c.endswith(metric_suffix)]
    if not cols_of_interest:
        print(f"No se encontraron columnas con el sufijo {metric_suffix}")
        return

    pos = {
        'Fp1': (-0.3, 0.9),  'Fp2': (0.3, 0.9),
        'F7':  (-0.8, 0.4),  'F8':  (0.8, 0.4),
        'T3':  (-1.0, 0.0),  'T4':  (1.0, 0.0),
        'T5':  (-0.8, -0.4), 'T6':  (0.8, -0.4),
        'O1':  (-0.3, -0.9), 'O2':  (0.3, -0.9),
        'F3':  (-0.4, 0.5),  'F4':  (0.4, 0.5),
        'C3':  (-0.5, 0.0),  'C4':  (0.5, 0.0),
        'P3':  (-0.4, -0.5), 'P4':  (0.4, -0.5),
        'Fz':  (0.0, 0.5),   'Cz':  (0.0, 0.0),  'Pz': (0.0, -0.5)
    }

    v_bckg  = df[df['macro_class'] == 'background'][cols_of_interest].median()
    v_focal = df[df['macro_class'] == 'focal'][cols_of_interest].median()
    v_gen   = df[df['macro_class'] == 'generalized'][cols_of_interest].median()

    x, y       = [], []
    z_b, z_f, z_g = [], [], []

    for col in cols_of_interest:
        pair = col.split('_')[0]
        try:
            e1, e2   = pair.split('-')
            e1_key   = next((k for k in pos.keys() if k.lower() == e1.lower()), None)
            e2_key   = next((k for k in pos.keys() if k.lower() == e2.lower()), None)
            if e1_key and e2_key:
                x.append((pos[e1_key][0] + pos[e2_key][0]) / 2)
                y.append((pos[e1_key][1] + pos[e2_key][1]) / 2)
                z_b.append(v_bckg[col])
                z_f.append(v_focal[col])
                z_g.append(v_gen[col])
        except Exception:
            continue

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor='white')

    vmin = min(min(z_b), min(z_f), min(z_g))
    vmax = max(max(z_b), max(z_f), max(z_g))
    xi, yi = np.meshgrid(np.linspace(-1.1, 1.1, 200), np.linspace(-1.1, 1.1, 200))

    titles    = ['Basal', 'Focal', 'Generalizada']
    datasets  = [z_b, z_f, z_g]
    cmap_choice = 'viridis' if 'ent' in metric_suffix else 'Spectral_r'

    for ax, data, title in zip(axes, datasets, titles):
        rbf = Rbf(x, y, data, function='multiquadric', smooth=0.1)
        zi  = rbf(xi, yi)
        zi[(xi ** 2 + yi ** 2) > 1.0] = np.nan

        im = ax.imshow(zi, extent=(-1.1, 1.1, -1.1, 1.1), origin='lower',
                       cmap=cmap_choice, vmin=vmin, vmax=vmax)

        ax.add_patch(Circle((0, 0), 1.0, color='black', fill=False, linewidth=1.2, zorder=10))
        ax.plot([-0.1, 0, 0.1], [1.0, 1.08, 1.0], color='black', linewidth=1.2, zorder=10)
        ax.add_patch(Ellipse((-1.02, 0), 0.12, 0.25, color='black', fill=False, linewidth=1.0))
        ax.add_patch(Ellipse((1.02, 0),  0.12, 0.25, color='black', fill=False, linewidth=1.0))

        ax.scatter(x, y, c='black', s=8, alpha=0.4, zorder=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.axis('off')

    plt.subplots_adjust(left=0.05, right=0.85, top=0.90, bottom=0.1, wspace=0.15)

    cbar_ax = fig.add_axes([0.88, 0.20, 0.015, 0.55])
    cbar    = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label, labelpad=20, weight='bold', fontsize=10, rotation=90)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    display(HTML(f'<div style="text-align: center;"><img src="data:image/png;base64,{data_uri}"></div>'))


def plot_kde(df_sample, feat1, feat2,
             xlabel1="Métrica 1", xlabel2="Métrica 2",
             ylabel="Frecuencia Relativa",
             target_col='macro_class', ylim=0.20):
    """
    Side-by-side KDE + histogram panels for two features with
    pairwise Mann-Whitney U effect sizes (Rosenthal r).

    Parameters
    ----------
    df_sample : pd.DataFrame
        Must contain `target_col` and the two feature columns.
    feat1 : str
        First feature column name.
    feat2 : str
        Second feature column name.
    xlabel1 : str
        X-axis label for the first panel.
    xlabel2 : str
        X-axis label for the second panel.
    ylabel : str
        Y-axis label (default 'Frecuencia Relativa').
    target_col : str
        Target column name (default 'macro_class').
    ylim : float
        Upper limit of the Y axis (default 0.20).
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2
    })
    sns.set_style("white")

    palette  = {'background': '#1f77b4', 'focal': '#d62728', 'generalized': '#2ca02c'}
    features = [feat1, feat2]
    x_labels = [xlabel1, xlabel2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, feature in enumerate(features):
        ax = axes[i]

        sns.histplot(
            data=df_sample, x=feature, hue=target_col, palette=palette,
            kde=True, element="step", stat="proportion", common_norm=False,
            alpha=0.2, ax=ax, line_kws={'linewidth': 2.5}
        )

        pairs = [('background', 'focal'), ('background', 'generalized'), ('focal', 'generalized')]
        p_values, eff_sizes = [], []

        for c1, c2 in pairs:
            g1 = df_sample[df_sample[target_col] == c1][feature].dropna()
            g2 = df_sample[df_sample[target_col] == c2][feature].dropna()
            n1, n2 = len(g1), len(g2)

            u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            p_values.append(p_val)

            mu_u    = (n1 * n2) / 2
            sigma_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
            z_score = (u_stat - mu_u) / sigma_u
            eff_sizes.append(abs(z_score) / np.sqrt(n1 + n2))

        _, p_adj, _, _ = multipletests(p_values, method='holm')

        labels_short = {'background': 'B', 'focal': 'F', 'generalized': 'G'}
        stats_text = ""
        for idx, (c1, c2) in enumerate(pairs):
            stars = "***" if p_adj[idx] < 0.001 else "**" if p_adj[idx] < 0.01 else "*" if p_adj[idx] < 0.05 else ""
            stats_text += f"{labels_short[c1]}-{labels_short[c2]}: {eff_sizes[idx]:.2f}{stars}\n"

        ax.text(0.05, 0.95, stats_text.strip(), transform=ax.transAxes,
                fontsize=11, verticalalignment='top', horizontalalignment='left',
                family='serif', fontweight='medium', linespacing=1.5)

        ax.set_xlabel(x_labels[i], fontsize=12, fontweight='bold', labelpad=12)
        ax.set_ylabel(ylabel if i == 0 else "", fontsize=12, fontweight='bold', labelpad=12)
        ax.set_ylim(0, ylim)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(True, linestyle=':', alpha=0.4)

        if i == 0:
            if ax.get_legend():
                ax.get_legend().remove()
        else:
            legend = ax.get_legend()
            if legend:
                labels_dict = {'background': 'Basal', 'focal': 'Focal', 'generalized': 'Generalizada'}
                for t in legend.get_texts():
                    t.set_text(labels_dict.get(t.get_text(), t.get_text()))
            sns.move_legend(ax, "upper right", bbox_to_anchor=(0.98, 0.98), title='', frameon=False, fontsize=10)

    plt.tight_layout()
    _display_figure(fig)


# =============================================================================
# SECTION 10 — SEIZURE WAVEFORM PREVIEW
# =============================================================================

def plot_seizure(parquet_path, csv_path, zoom_window=10, pre_seconds=3):
    """
    Plot the onset of a seizure across three EEG channels.

    Parameters
    ----------
    parquet_path : str
        Path to the Parquet file with EEG signal data.
    csv_path : str
        Path to the CSV annotation file for that session.
    zoom_window : int
        Total number of seconds to display (default 10).
    pre_seconds : int
        Seconds of pre-ictal signal to include before seizure onset (default 3).
    """
    df_sig = pd.read_parquet(parquet_path)
    df_ann = pd.read_csv(csv_path, comment='#', skipinitialspace=True)

    ictal_events = df_ann[df_ann['label'] != 'bckg']
    if ictal_events.empty:
        return None

    seizure_start = ictal_events['start_time'].min()
    seizure_label = ictal_events['label'].iloc[0].upper()

    t_plot_start = max(0, seizure_start - pre_seconds)
    t_plot_end   = t_plot_start + zoom_window

    fs      = 250
    df_plot = df_sig.iloc[int(t_plot_start * fs):int(t_plot_end * fs)].copy()
    channels = df_plot.columns[:3]
    df_plot  = df_plot[channels]

    time_axis = np.linspace(t_plot_start, t_plot_end, len(df_plot))

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')

    for i, col in enumerate(channels):
        sig = df_plot[col].values
        if np.ptp(sig) == 0:
            continue
        sig_norm = (sig - np.mean(sig)) / np.ptp(sig)
        ax.plot(time_axis, sig_norm - i, color='black', linewidth=1.0, alpha=0.9)

    ax.axvline(seizure_start, color='#d62728', linestyle='--', linewidth=1.8, alpha=0.8)
    ax.axvspan(seizure_start, t_plot_end, color='#d62728', alpha=0.06)

    ax.text(seizure_start + 0.15, 0.04, f'INICIO {seizure_label}', color='black', fontsize=9,
            rotation=0, transform=ax.get_xaxis_transform(), va='bottom')
    ax.text(0.92, 0.95, 'Paciente 1 - Sesión 2', color='black', fontsize=10,
            transform=ax.transAxes, ha='right', va='top')

    ax.set_yticks([-i for i in range(len(channels))])
    ax.set_yticklabels(channels, fontsize=10, fontweight='bold')
    ax.set_xlabel('Tiempo (segundos)', fontsize=10, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')
    display(HTML(f'''
        <div style="text-align: center; margin: 15px 0;">
            <img src="data:image/png;base64,{data_uri}" style="max-width: 75%; border: none;">
        </div>
    '''))


# =============================================================================
# SECTION 11 — TRAINING SET AUDIT
# =============================================================================

def get_train_audit(csv_path, parquet_path, metadata_path, global_collapse=True):
    """
    Compute natural (raw CSV) and processed (windowed Parquet) class distributions
    for training-set patients.

    Parameters
    ----------
    csv_path : str
        Root directory with per-session annotation CSV files.
    parquet_path : str
        Root directory with per-session feature Parquet files.
    metadata_path : str
        Path to the Parquet metadata file
        (training patients have split_final == 0).
    global_collapse : bool
        If True, map specific seizure types to macro-classes (default True).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_natural, df_processed) — each indexed by label with
        duration/count and proportion columns.
    """
    try:
        df_meta = pd.read_parquet(metadata_path)
        train_patients = set(df_meta[df_meta['split_final'] == 0]['patient_num_id'].astype(str))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    mapping = {
        'fnsz': 'focal',   'cpsz': 'focal',        'spsz': 'focal',
        'gnsz': 'generalized', 'absz': 'generalized', 'tcsz': 'generalized',
        'tnsz': 'generalized', 'mysz': 'generalized', 'combined': 'generalized',
        'bckg': 'background'
    }

    all_csvs     = list(Path(csv_path).rglob("*.csv"))
    csv_files    = [f for f in all_csvs    if any(part in train_patients for part in f.parts)]

    all_parquets  = list(Path(parquet_path).rglob("*.parquet"))
    parquet_files = [f for f in all_parquets if any(part in train_patients for part in f.parts)]

    # Natural durations
    durations = defaultdict(float)
    for f in csv_files:
        try:
            df = pd.read_csv(f, comment='#', skipinitialspace=True)
            unique_intervals = df[['start_time', 'stop_time', 'label']].drop_duplicates()
            for _, row in unique_intervals.iterrows():
                lbl = row['label'].lower()
                final_lbl = mapping.get(lbl, lbl) if global_collapse else lbl
                durations[final_lbl] += (row['stop_time'] - row['start_time'])
        except Exception:
            continue

    # Windowed counts
    windows_count = defaultdict(int)
    WINDOW_SEC = 4.096

    for f in parquet_files:
        try:
            table  = pq.read_table(f, columns=['label'])
            labels = table.column('label').to_pandas().str.strip().str.lower()
            if global_collapse:
                labels = labels.map(lambda x: mapping.get(x, x))
            counts = labels.value_counts()
            for lbl, count in counts.items():
                windows_count[lbl] += count
        except Exception:
            continue

    def build_summary(data_dict, val_name, is_processed=False):
        if not data_dict:
            return pd.DataFrame(columns=['Label', val_name, 'Prop (%)']).set_index('Label')

        df = pd.DataFrame([{'Label': k, val_name: v} for k, v in data_dict.items()])

        if is_processed:
            df['Segundos']  = df[val_name] * WINDOW_SEC
            total_time      = df['Segundos'].sum()
            df['Prop (%)']  = (df['Segundos'] / total_time) * 100
        else:
            total_time      = df[val_name].sum()
            df['Prop (%)']  = (df[val_name] / total_time) * 100

        return df.set_index('Label').sort_values(by='Prop (%)', ascending=False)

    df_nat  = build_summary(durations,     'Segundos')
    df_proc = build_summary(windows_count, 'Ventanas', is_processed=True)

    return df_nat, df_proc


def plot_train_comparison(df_natural, df_processed):
    """
    Symmetric two-panel audit figure comparing natural vs. windowed distributions.

    Left panel  — Natural CSV distribution.
    Right panel — Processed (windowed) distribution.

    Both panels share the same primary Y axis (proportion %) and have a
    secondary Y axis (absolute seconds). Bars are annotated with hours.

    Parameters
    ----------
    df_natural : pd.DataFrame
        Output from get_train_audit() — natural distribution.
    df_processed : pd.DataFrame
        Output from get_train_audit() — processed distribution.
    """
    LABEL_MAPPING = {'background': 'Basal', 'focal': 'Focal', 'generalized': 'Generalizada'}
    PALETTE       = {'Basal': '#1f77b4', 'Focal': '#d62728', 'Generalizada': '#2ca02c'}
    WINDOW_DURATION = 4.096

    def prepare_audit_data(df):
        df_copy = df.copy()
        df_copy.index = df_copy.index.map(lambda x: LABEL_MAPPING.get(x.lower(), x.capitalize()))

        if 'Ventanas' in df_copy.columns:
            df_copy['Segundos'] = df_copy['Ventanas'] * WINDOW_DURATION
            total_seconds       = df_copy['Segundos'].sum()
            df_copy['Prop (%)'] = (df_copy['Segundos'] / total_seconds) * 100

        return df_copy

    df_nat_clean  = prepare_audit_data(df_natural)
    df_proc_clean = prepare_audit_data(df_processed)

    max_prop_limit = max(df_nat_clean['Prop (%)'].max(), df_proc_clean['Prop (%)'].max()) * 1.18
    max_secs_limit = max(df_nat_clean['Segundos'].max(), df_proc_clean['Segundos'].max()) * 1.18

    plt.rcParams.update({
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5
    })
    sns.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    plt.subplots_adjust(wspace=0.08)

    datasets = [df_nat_clean, df_proc_clean]
    titles   = ["Distribución Natural", "Estrategia de Ventaneo"]

    for i, ax in enumerate(axes):
        current_df = datasets[i]
        colors     = [PALETTE.get(label, '#7f8c8d') for label in current_df.index]

        bars = ax.bar(current_df.index, current_df['Prop (%)'],
                      color=colors, edgecolor='black', alpha=0.8, width=0.6)

        ax.set_title(titles[i], fontsize=11, fontweight='bold', pad=14)
        ax.set_ylim(0, max_prop_limit)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel("")

        if i == 0:
            ax.set_ylabel("Proporción del Dataset (%)", fontsize=10, fontweight='bold')
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        ax_sec = ax.twinx()
        ax_sec.set_ylim(0, max_secs_limit)

        if i == 1:
            ax_sec.set_ylabel("Magnitud Temporal (Seg)", fontsize=10, fontweight='bold', labelpad=12)
            ax_sec.tick_params(axis='y', labelsize=9)
        else:
            ax_sec.tick_params(axis='y', which='both', right=False, labelright=False)

        for bar, label in zip(bars, current_df.index):
            height         = bar.get_height()
            duration_hours = current_df.loc[label, 'Segundos'] / 3600

            ax.text(bar.get_x() + bar.get_width() / 2, height + (max_prop_limit * 0.02),
                    f'{duration_hours:,.1f} h', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='black')

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    _display_figure(fig)


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _display_figure(fig, dpi=150, max_width=None):
    """Encode a matplotlib figure as base64 PNG and display it centred in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    data_uri = base64.b64encode(buf.getvalue()).decode('utf-8')

    style = "border: none;"
    if max_width:
        style += f" max-width: {max_width}; height: auto;"

    display(HTML(
        f'<div style="text-align: center; padding: 10px;">'
        f'<img src="data:image/png;base64,{data_uri}" style="{style}">'
        f'</div>'
    ))