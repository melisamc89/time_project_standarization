#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SI and Decoder Grouped Boxplots (Deep vs. Superficial) + OLS ANOVA with post-hoc tests.

Reads precomputed tables from cluster_umap_si_decoder_with_all_cells_and_mse.
Generates grouped boxplots with jittered points and exports TXT stats for:
  (a) all datasets pooled,
  (b) each individual dataset,
  (c) lt_rot LT-only and ROT-only split by deep/superficial.
"""

from pathlib import Path
from itertools import combinations
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import Rectangle

# =============================================================================
# Configuration & Styling
# =============================================================================

ROOT = Path(
    '/home/hippolab/Documents/melisa/time_project_standarization/analysis/cluster_si_dual_space_outlier_filtered')
SI_FILE = ROOT / 'tables/cluster_structure_index_dual_space.csv'
DEC_FILE = ROOT / 'tables/cluster_decoder_xgboost_long.csv'
OUTPUT = ROOT / 'figures_grouped_boxplots_dual_space'

DEPTH_COLOR_MAP = {
    'superficial': '#9900ff',
    'deep': '#cc9900',
    'whole population': '#888888',
    'whole_population': '#888888',
    'whole': '#888888',
    'all': '#888888',
    'unknown': '#888888',
}

GROUP_ORDER = [-1, 2, 0, 1]
GROUP_NAMES = {-1: 'All cells', 2: 'Spatial', 0: 'Temporal', 1: 'Time × dir'}
GROUP_SWATCH = {-1: '#888888', 2: '#df8ce7', 0: '#a6d989', 1: '#62c5e4'}

FIG_DPI = 300
SHOW_SIGNIFICANCE = True
MAX_BRACKETS = 4
ALPHA = 0.05
SHOW_POINTS = True


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(s))


def depth_normalize(s):
    s = str(s).strip().lower()
    return {'sup': 'superficial', 'whole_population': 'whole population',
            'whole': 'whole population', 'all': 'whole population'}.get(s, s)


# =============================================================================
# Data Loading & Scoping
# =============================================================================

def load_tables():
    for path in (SI_FILE, DEC_FILE):
        if not path.is_file():
            raise FileNotFoundError(f'Input table missing: {path}')
    si = pd.read_csv(SI_FILE)
    de = pd.read_csv(DEC_FILE)
    for frame, value in ((si, 'si_value'), (de, 'metric_value')):
        needed = {'dataset_name', 'mouse_name', 'session_name', 'depth_type',
                  'group_id', 'behavior', value}
        absent = needed - set(frame.columns)
        if absent:
            raise ValueError(f'Missing columns {absent} in {value} table')
        frame['group_id'] = pd.to_numeric(frame['group_id'], errors='coerce')
        frame['depth'] = frame['depth_type'].map(depth_normalize)
        frame['group'] = frame['group_id'].map(GROUP_NAMES)
        frame[value] = pd.to_numeric(frame[value], errors='coerce')
        frame['dataset_name'] = frame['dataset_name'].astype(str)
        frame['mouse_name'] = frame['mouse_name'].astype(str)
        frame['session_name'] = frame['session_name'].astype(str)
    if 'si_space' not in si.columns:
        raise ValueError('SI table must contain si_space')
    if 'metric_name' not in de.columns:
        raise ValueError('Decoder table must have metric_name')
    return si, de


def get_scopes(df):
    yield 'all_datasets', df
    for dname in sorted(df['dataset_name'].unique()):
        yield f'dataset_{safe_name(dname)}', df[df['dataset_name'] == dname]
    ltrot = df[df['dataset_name'].str.lower() == 'lt_rot']
    if not ltrot.empty:
        for mode in ('lt', 'rot'):
            part = ltrot[ltrot['session_name'].str.lower().str.contains(mode, regex=False, na=False)]
            part = part[part['depth'].isin(['deep', 'superficial'])]
            if not part.empty:
                yield f'lt_rot_{mode}_only', part


def finite_df(df, value):
    out = df[df['group_id'].isin(GROUP_ORDER)].copy()
    return out[np.isfinite(out[value].to_numpy(dtype=float))]


# =============================================================================
# Statistics (Ordinary OLS ANOVA + Holm-adjusted Welch t-tests)
# =============================================================================

def param_stats(frame, factors, value):
    d = frame.copy()
    d['_value'] = pd.to_numeric(d[value], errors='coerce')
    d = d[np.isfinite(d['_value'])]
    cols = ['term', 'df_num', 'df_den', 'F', 'p', 'partial_eta2',
            'SS', 'MS', 'SS_error', 'MS_error', 'method']
    if len(d) < 6 or any(d[f].nunique() < 2 for f in factors):
        return pd.DataFrame(columns=cols), 'ANOVA skipped: insufficient data or factor levels.'
    if len(factors) == 2:
        observed = d.groupby(factors, observed=True).size()
        expected = d[factors[0]].nunique() * d[factors[1]].nunique()
        if len(observed) < expected:
            return pd.DataFrame(columns=cols), 'Two-way ANOVA skipped: missing cell combinations.'
    formula = '_value ~ ' + ' * '.join(f'C({f})' for f in factors)
    try:
        fit = ols(formula, data=d).fit()
        if fit.df_resid <= 0:
            return pd.DataFrame(columns=cols), 'ANOVA skipped: no residual degrees of freedom.'
        aov = sm.stats.anova_lm(fit, typ=2)
        se = float(aov.loc['Residual', 'sum_sq'])
        de = float(aov.loc['Residual', 'df'])
        mse = se / de
        rows = []
        for term, rec in aov.iterrows():
            if term == 'Residual':
                continue
            ss = float(rec['sum_sq'])
            df_num = float(rec['df'])
            rows.append({
                'term': term, 'df_num': df_num, 'df_den': de,
                'F': float(rec['F']), 'p': float(rec['PR(>F)']),
                'partial_eta2': ss / (ss + se) if ss + se > 0 else np.nan,
                'SS': ss, 'MS': ss / df_num, 'SS_error': se,
                'MS_error': mse, 'method': 'ordinary OLS ANOVA Type II',
            })
        description = (
            f'Ordinary {len(factors)}-factor OLS ANOVA, Type-II sums of squares; '
            f'N={len(d)} observations, residual df={de:g}.'
        )
        return pd.DataFrame(rows, columns=cols), description
    except Exception as exc:
        return pd.DataFrame(columns=cols), f'ANOVA failed: {exc}'


def pairwise_tests(df, value, factor, by=None):
    data = []
    partitions = [(None, df)] if by is None else list(df.groupby(by, observed=True))
    for split_level, part in partitions:
        levels = list(pd.unique(part[factor].dropna()))
        if factor == 'group':
            levels = [GROUP_NAMES[x] for x in GROUP_ORDER if GROUP_NAMES[x] in levels]
        for a, b in combinations(levels, 2):
            v1 = part.loc[part[factor] == a, value].dropna().to_numpy(dtype=float)
            v2 = part.loc[part[factor] == b, value].dropna().to_numpy(dtype=float)
            if min(len(v1), len(v2)) < 2:
                continue
            stat, p = stats.ttest_ind(v1, v2, equal_var=False)
            q1, q2 = np.var(v1, ddof=1) / len(v1), np.var(v2, ddof=1) / len(v2)
            denom = q1 ** 2 / (len(v1) - 1) + q2 ** 2 / (len(v2) - 1)
            dfree = (q1 + q2) ** 2 / denom if denom > 0 else np.nan
            data.append({
                'stratum': 'all' if split_level is None else str(split_level),
                'factor': factor, 'group_a': a, 'group_b': b,
                'test': 'independent Welch t-test',
                'n_a': len(v1), 'n_b': len(v2), 'statistic': stat,
                'df': dfree, 'p_raw': p,
                'mean_difference': float(np.mean(v1) - np.mean(v2)),
            })
    result = pd.DataFrame(data)
    if not result.empty:
        good = np.isfinite(result['p_raw'])
        result['p_holm'] = np.nan
        if good.any():
            result.loc[good, 'p_holm'] = multipletests(result.loc[good, 'p_raw'], method='holm')[1]
    return result


def run_stats(df, value, split):
    factors = ['group', 'depth'] if split else ['group']
    anova, note = param_stats(df, factors, value)
    posts = []
    if split:
        posts.append(pairwise_tests(df, value, 'group', by='depth'))
        posts.append(pairwise_tests(df, value, 'depth', by='group'))
    else:
        posts.append(pairwise_tests(df, value, 'group'))
    posts = pd.concat([p for p in posts if not p.empty], ignore_index=True) if any(
        not p.empty for p in posts) else pd.DataFrame()
    return anova, posts, note


# =============================================================================
# Plotting & Annotations
# =============================================================================

def annotate_top_significant(ax, sub, posts, split):
    if not SHOW_SIGNIFICANCE or posts.empty:
        return
    use = posts[(posts['factor'] == 'group') & (posts['p_holm'] < ALPHA)].copy()
    if use.empty:
        return
    use = use.sort_values('p_holm').head(MAX_BRACKETS)
    indices = {GROUP_NAMES[g]: i for i, g in enumerate(GROUP_ORDER)}
    vals = sub['_value'].to_numpy()
    lo, hi = np.min(vals), np.max(vals)
    spread = max(hi - lo, 0.1)

    for number, (_, row) in enumerate(use.iterrows()):
        xa, xb = indices.get(row['group_a']), indices.get(row['group_b'])
        if xa is None or xb is None:
            continue
        offset = 0.20 if split else 0
        if split:
            offset = -0.20 if row['stratum'] == 'deep' else +0.20
        xa += offset
        xb += offset
        y = hi + (number + 1) * 0.10 * spread
        ax.plot([xa, xa, xb, xb], [y - 0.025 * spread, y, y, y - 0.025 * spread],
                color='black', linewidth=1.1, clip_on=False)
        p = row['p_holm']
        star = '***' if p < .001 else '**' if p < .01 else '*'
        ax.text((xa + xb) / 2, y + 0.015 * spread, star, ha='center', fontsize=10)
    ax.set_ylim(lo - .08 * spread, hi + (len(use) + 1) * .13 * spread)


def make_grouped_boxplot(df, value, title, out_base, split_requested):
    d = finite_df(df, value)
    if d.empty:
        return
    d['_value'] = d[value]
    order = [GROUP_NAMES[g] for g in GROUP_ORDER]
    d['group'] = pd.Categorical(d['group'], categories=order, ordered=True)

    depths = set(d['depth'].unique())
    split = split_requested and {'deep', 'superficial'}.issubset(depths)
    if split:
        d = d[d['depth'].isin(['deep', 'superficial'])].copy()

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    if split:
        # Grouped boxplot: Deep (#cc9900) vs. Superficial (#9900ff) side-by-side
        hue_order = ['deep', 'superficial']
        palette = {k: DEPTH_COLOR_MAP[k] for k in hue_order}

        sns.boxplot(
            data=d, x='group', y='_value', hue='depth', order=order,
            hue_order=hue_order, palette=palette, width=0.6,
            showcaps=True, boxprops=dict(linewidth=1.3),
            whiskerprops=dict(linewidth=1.3), flierprops=dict(marker=''),
            ax=ax
        )
        if SHOW_POINTS:
            sns.stripplot(
                data=d, x='group', y='_value', hue='depth', order=order,
                hue_order=hue_order, dodge=True, color='black', alpha=0.65,
                size=4.5, jitter=0.2, ax=ax, legend=False
            )
        ax.legend(title='Depth', frameon=False, loc='upper right')
    else:
        # Single boxplot per group
        palette = {GROUP_NAMES[g]: GROUP_SWATCH[g] for g in GROUP_ORDER}
        sns.boxplot(
            data=d, x='group', y='_value', order=order,
            palette=palette, hue='group', legend=False, width=0.55,
            showcaps=True, boxprops=dict(linewidth=1.3),
            whiskerprops=dict(linewidth=1.3), flierprops=dict(marker=''),
            ax=ax
        )
        if SHOW_POINTS:
            sns.stripplot(
                data=d, x='group', y='_value', order=order,
                color='black', alpha=0.65, size=4.5, jitter=0.2, ax=ax
            )

    anova, post, note = run_stats(d, value, split)
    annotate_top_significant(ax, d, post, split)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Structure Index' if value == 'si_value' else title.split(' | ')[-1])
    ax.tick_params(axis='x', rotation=35)

    # Cluster-color swatches under the x-axis tick labels
    for i, gid in enumerate(GROUP_ORDER):
        ax.add_patch(Rectangle((i - 0.17, -0.23), 0.34, 0.045,
                               transform=ax.get_xaxis_transform(),
                               facecolor=GROUP_SWATCH[gid], edgecolor='none',
                               clip_on=False))

    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix('.png'), dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Save full statistical report
    description = d.groupby(['group', 'depth'], observed=True)['_value'].agg(
        ['count', 'mean', 'std', 'median']).to_string()
    with open(out_base.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{title}\n' + '=' * len(title) + '\n\n')
        f.write('ANALYSIS DESIGN\n')
        f.write(f"Factors: {', '.join(['group', 'depth'] if split else ['group'])}\n")
        f.write(f'{note}\n')
        f.write(
            'CAVEAT: multiple observations per animal/session; ordinary ANOVA ignores this correlation structure.\n\n')
        f.write('DESCRIPTIVE STATISTICS\n' + description + '\n\n')
        f.write('ANOVA (effect df_num, error df_den, F, p, partial eta², SS if available)\n')
        f.write(anova.to_string(index=False, float_format=lambda n: f'{n:.6g}') + '\n\n')
        f.write('POST-HOC COMPARISONS (Holm correction within this figure)\n')
        f.write(post.to_string(index=False, float_format=lambda n: f'{n:.6g}') if not post.empty else 'Not available')
        f.write('\n')


# =============================================================================
# Main
# =============================================================================

def main():
    si, dec = load_tables()
    planned = 0
    for kind, data, value in [('si', si, 'si_value'), ('decoder', dec, 'metric_value')]:
        for scope, frame in get_scopes(data):
            split_requested = (scope != 'all_datasets' and
                               {'deep', 'superficial'}.issubset(set(frame['depth'])))
            groupings = ['si_space', 'behavior'] if kind == 'si' else ['behavior', 'metric_name']
            for labels, part in frame.groupby(groupings, sort=True):
                labels = labels if isinstance(labels, tuple) else (labels,)
                space = str(labels[0]) if kind == 'si' else 'neural_decoder'
                behavior = str(labels[1]) if kind == 'si' else str(labels[0])
                metric = '' if kind == 'si' else str(labels[1])
                title = f'{scope} | {space} | {behavior} | {"SI" if kind == "si" else metric}'
                path = OUTPUT / kind / safe_name(scope) / safe_name(space) / (
                    '__'.join([safe_name(behavior)] + ([safe_name(metric)] if metric else [])))
                try:
                    make_grouped_boxplot(part, value, title, path, split_requested)
                    planned += 1
                except Exception as exc:
                    print(f'[ERROR] {title}: {exc}')
    print(f'Completed {planned} boxplot groups. Output: {OUTPUT}')


if __name__ == '__main__':
    main()
