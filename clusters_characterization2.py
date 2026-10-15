#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SI and decoder violin plots + one/two-factor ANOVA with post-hoc tests.

Reads precomputed tables from cluster_umap_si_decoder_with_all_cells_and_mse.
Figures + TXT stats for (a) all datasets pooled, (b) each dataset,
(c) lt_rot LT-only and ROT-only split by deep/superficial.

Uses only ordinary OLS ANOVA (no repeated-measures models):
  one-way: group
  two-way: group × depth including interaction.
Post-hoc: pairwise independent-sample Welch t-tests with Holm correction.
Caution: observations from the same session may not be independent.
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

ROOT = Path(
    '/home/hippolab/Documents/melisa/time_project_standarization/analysis/cluster_umap_si_decoder_with_all_cells_and_mse')
SI_FILE = ROOT / 'tables/cluster_structure_index.csv'
DEC_FILE = ROOT / 'tables/cluster_decoder_xgboost_long.csv'
OUTPUT = ROOT / 'figures_violin_anova'

DEPTH_COLOR_MAP = {
    'superficial': '#9900ff',
    'deep': '#cc9900',
    'whole population': 'gray',
    'whole_population': 'gray',
    'whole': 'gray',
    'all': 'gray',
    'unknown': 'gray',
}
GROUP_ORDER = [-1, 2, 0, 1]
GROUP_NAMES = {-1: 'All cells', 2: 'Spatial', 0: 'Temporal', 1: 'Time × dir'}
GROUP_SWATCH = {-1: '#888888', 2: '#df8ce7', 0: '#a6d989', 1: '#62c5e4'}

FIG_DPI = 300
SHOW_SIGNIFICANCE = True
MAX_BRACKETS = 4
ALPHA = 0.05
SPLIT_ONLY_WHEN_BOTH_DEPTHS = True
SHOW_POINTS = False


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(s))


def depth_normalize(s):
    s = str(s).strip().lower()
    return {'sup': 'superficial', 'whole_population': 'whole population',
            'whole': 'whole population', 'all': 'whole population'}.get(s, s)


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
        # Subject for within-session, within-depth comparisons
        frame['subject_depth'] = (frame['dataset_name'] + '|' + frame['mouse_name'] + '|' +
                                  frame['session_name'] + '|' + frame['depth'])
        # Subject for testing group and depth together (matched layers)
        frame['subject_session'] = (frame['dataset_name'] + '|' + frame['mouse_name'] + '|' +
                                    frame['session_name'])
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


def param_stats(frame, factors, value):
    """Ordinary OLS ANOVA with Type-II sums of squares, F, df, p, eta_p²."""
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
            return pd.DataFrame(columns=cols), ('Two-way ANOVA skipped: '
                                                'at least one group × depth combination is missing.')
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
            f'N={len(d)} observations, residual df={de:g}. '
            'This model treats rows as independent (does not model animal/session dependence). '
            'Post-hoc comparisons use Welch t-tests with Holm correction.'
        )
        return pd.DataFrame(rows, columns=cols), description
    except Exception as exc:
        return pd.DataFrame(columns=cols), f'ANOVA failed: {exc}'


def pairwise_tests(df, value, factor, by=None):
    """Independent-samples Welch comparisons and Holm adjustment."""
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
            result.loc[good, 'p_holm'] = multipletests(
                result.loc[good, 'p_raw'], method='holm')[1]
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


def annotate_top_significant(ax, sub, posts, split):
    if not SHOW_SIGNIFICANCE or posts.empty:
        return
    # For split violins annotate group comparisons only when significant.
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
        offset = 0.13 if split else 0
        if split:
            offset = -0.13 if row['stratum'] == 'deep' else +0.13
        xa += offset
        xb += offset
        y = hi + (number + 1) * 0.10 * spread
        ax.plot([xa, xa, xb, xb], [y - 0.025 * spread, y, y, y - 0.025 * spread],
                color='black', linewidth=1.1, clip_on=False)
        p = row['p_holm']
        star = '***' if p < .001 else '**' if p < .01 else '*'
        ax.text((xa + xb) / 2, y + 0.015 * spread, star, ha='center', fontsize=10)
    ax.set_ylim(lo - .08 * spread, hi + (len(use) + 1) * .13 * spread)


def make_plot(df, value, title, out_base, split_requested):
    d = finite_df(df, value)
    if d.empty:
        return
    d['_value'] = d[value]
    d['group'] = pd.Categorical(d['group'],
                                categories=[GROUP_NAMES[g] for g in GROUP_ORDER],
                                ordered=True)
    depths = set(d['depth'].unique())
    split = split_requested and {'deep', 'superficial'}.issubset(depths)
    if split:
        d = d[d['depth'].isin(['deep', 'superficial'])].copy()
    order = [GROUP_NAMES[g] for g in GROUP_ORDER]
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    if split:
        sns.violinplot(data=d, x='group', y='_value', hue='depth', order=order,
                       hue_order=['deep', 'superficial'], split=True,
                       palette={k: DEPTH_COLOR_MAP[k] for k in ['deep', 'superficial']},
                       inner=None, cut=0, linewidth=1.1, density_norm='width', ax=ax)
        ax.legend(title='Depth', frameon=False, loc='upper right')
    else:
        sns.violinplot(data=d, x='group', y='_value', order=order,
                       palette={GROUP_NAMES[g]: GROUP_SWATCH[g] for g in GROUP_ORDER},
                       hue='group', legend=False, inner=None, cut=0,
                       linewidth=1.1, density_norm='width', ax=ax)
    anova, post, note = run_stats(d, value, split)
    annotate_top_significant(ax, d, post, split)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Structure Index' if value == 'si_value' else title.split(' | ')[-1])
    ax.tick_params(axis='x', rotation=35)
    # Small cluster-color swatches under the tick labels (reference figure).
    from matplotlib.patches import Rectangle
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

    description = d.groupby(['group', 'depth'], observed=True)['_value'].agg(
        ['count', 'mean', 'std', 'median']).to_string()
    with open(out_base.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{title}\n' + '=' * len(title) + '\n\n')
        f.write('ANALYSIS DESIGN\n')
        f.write(f"Factors: {', '.join(['group', 'depth'] if split else ['group'])}\n")
        f.write(f'{note}\n')
        f.write(
            'CAVEAT: the same animal/session may contribute several groups; ordinary ANOVA ignores this dependency.\n')
        f.write('No multiple-comparison correction across distinct behaviors/metrics.\n\n')
        f.write('DESCRIPTIVE STATISTICS\n' + description + '\n\n')
        f.write('ANOVA (effect df_num, error df_den, F, p, partial eta², SS if available)\n')
        f.write(anova.to_string(index=False, float_format=lambda n: f'{n:.6g}') + '\n\n')
        f.write('POST-HOC COMPARISONS (Holm correction within this figure)\n')
        f.write(post.to_string(index=False, float_format=lambda n: f'{n:.6g}') if not post.empty else 'Not available')
        f.write('\n')


def make_reference_boxplot(df, value, title, out_base):
    """Four group-colored boxes, observations, and Holm-adjusted post-hoc brackets.

    This is a GROUP-only (one-factor) view; for depth effects and interactions,
    consult the split-violin plot and its two-way ANOVA report.
    """
    d = finite_df(df, value).copy()
    if d.empty:
        return
    d['_value'] = d[value]
    ordered = [GROUP_NAMES[g] for g in GROUP_ORDER]
    d['group'] = pd.Categorical(d['group'], categories=ordered, ordered=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rng = np.random.default_rng(42)
    for i, gid in enumerate(GROUP_ORDER):
        label = GROUP_NAMES[gid]
        values = d.loc[d['group'] == label, '_value'].dropna().to_numpy()
        if len(values) == 0:
            continue
        color = GROUP_SWATCH[gid]
        ax.boxplot([values], positions=[i], widths=0.58, patch_artist=True,
                   showfliers=False,
                   boxprops=dict(facecolor=color, edgecolor='black', linewidth=1.5),
                   medianprops=dict(color='black', linewidth=1.6),
                   whiskerprops=dict(color='black', linewidth=1.2),
                   capprops=dict(color='black', linewidth=1.2))
        ax.scatter(i + rng.uniform(-0.11, 0.11, len(values)), values,
                   c='black', s=17, zorder=5, alpha=.85)
    ax.set_xticks(range(len(GROUP_ORDER)), ordered, rotation=90)
    ax.set_xlim(-0.55, len(GROUP_ORDER) - 0.45)
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Structure Index' if value == 'si_value' else title.split(' | ')[-1])
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)

    # One-factor post-hoc brackets across groups. Use independent Welch tests (not paired).
    anova, post, note = run_stats(d, value, split=False)
    annotate_top_significant(ax, d, post, split=False)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix('.png'), dpi=FIG_DPI,
                bbox_inches='tight', facecolor='white')
    fig.savefig(out_base.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    desc = d.groupby('group', observed=True)['_value'].agg(['count', 'mean', 'std', 'median'])
    with open(out_base.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        f.write(title + '\n\nONE-FACTOR GROUP COMPARISON (pooled over depths if present)\n')
        f.write(note + '\n\nDESCRIPTIVES\n' + desc.to_string() + '\n\n')
        f.write('ANOVA (df numerator / denominator; F; p; partial eta²)\n')
        f.write(anova.to_string(index=False) + '\n\n')
        f.write('POST-HOC (Holm-adjusted)\n')
        f.write(post.to_string(index=False) if not post.empty else 'Not available')
        f.write('\n')


def main():
    si, dec = load_tables()
    planned = 0
    for kind, data, value in [('si', si, 'si_value'), ('decoder', dec, 'metric_value')]:
        for scope, frame in get_scopes(data):
            # Split LT/ROT and any single dataset with both Deep and Superficial.
            # For all-datasets pooled, one-way across groups (not dataset/depth).
            split_requested = scope != 'all_datasets' and {'deep', 'superficial'}.issubset(set(frame['depth']))
            groupings = ['behavior'] if kind == 'si' else ['behavior', 'metric_name']
            for labels, part in frame.groupby(groupings, sort=True):
                labels = labels if isinstance(labels, tuple) else (labels,)
                behavior = str(labels[0])
                metric = '' if kind == 'si' else str(labels[1])
                title = f'{scope} | {behavior} | {"SI" if kind == "si" else metric}'
                path = OUTPUT / kind / safe_name(scope) / (
                    '__'.join([safe_name(behavior)] + ([safe_name(metric)] if metric else [])))
                try:
                    make_plot(part, value, title, path, split_requested)
                    make_reference_boxplot(part, value, title + ' | boxplot',
                                           path.parent / 'boxplots' / path.name)
                    planned += 1
                except Exception as exc:
                    print(f'[ERROR] {title}: {exc}')
    print(f'Completed {planned} plot/stat groups. Output: {OUTPUT}')


if __name__ == '__main__':
    main()
