#!/usr/bin/env python3
# =============================================================================
# campaign_axis_audit.py
#
# READ-ONLY auditor for a cohort of sweep campaigns produced by
# HPC_main_sweep.py.  It answers, per campaign and across campaigns:
#
#   (1) WHICH PARAMETERS WERE SWEPT -- declared (job_args.json /
#       manifest.json 'axis_declaration') AND verified empirically by scanning
#       a random sample of iter_*.npz 'params' columns.  Astrocyte axes are
#       reported explicitly, including whether they were CONSUMED (mode=Full)
#       or merely drawn while nothing read them (mode=Neuronal -> INERT).
#   (2) WHICH BOUNDS each campaign used (manifest 'param_bounds'), aligned BY
#       AXIS NAME, so campaigns whose registry has a different dimensionality
#       (e.g. 36-D without O_N vs 37-D with O_N) are still comparable.
#   (3) THE SEED FOOTPRINT of every campaign (_resolved_seed_master per task),
#       with within-campaign reuse and PAIRWISE CROSS-CAMPAIGN OVERLAP -- i.e.
#       a direct measurement of the seed-collision bug.
#   (4) THE CULTURE GEOMETRY actually used (c_max, Nn, Na, density,
#       density_astro, topology_mode, conn_rule, conn_periodic) and the
#       connectivity prior actually drawn (conn_prob range under 'flat';
#       p0/d0/beta ranges under 'weibull').
#
# WHY BOTH DECLARED AND EMPIRICAL
#   The declaration says what the launcher intended; the npz columns say what
#   the sampler actually did.  They can disagree in two ways that matter:
#     * an axis declared swept whose column is constant (bounds were a point
#       interval, or a freeze override landed after the declaration), and
#     * an axis that varies but is causally inert (drawn, recorded, read by
#       nothing).  A variance scan alone CANNOT detect the second case -- the
#       column varies perfectly well -- which is exactly why the declaration
#       is read as well and the two are cross-tabulated.
#
# GUARANTEES
#   * Read-only.  Nothing under the campaign tree is created, modified or
#     deleted.  All output goes to --out (default ./campaign_axis_audit).
#   * stdlib + numpy only.  No Brian2, no scipy, no matplotlib, no import of
#     HPC_main_sweep.py / HPC_single_run.py (their registries are read from
#     SOURCE with ast, never executed).
#   * manifest.json files are read HEAD-ONLY (everything before the huge
#     'topologies' array), so a multi-hundred-MB manifest costs a few KB.
#
# USAGE (on the cluster login node; no allocation needed)
#   python3 campaign_axis_audit.py /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro \
#       --glob 'campaign_cadex_hhgap_v*' \
#       --registry-src /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/HPC_single_run.py \
#       --sweep-src    /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/HPC_main_sweep.py \
#       --out ./audit_hhgap
#
#   Metadata only (seconds, no npz reads):
#       python3 campaign_axis_audit.py <ROOT> --quick
#
# OUTPUTS (under --out)
#   campaign_axis_audit.md     human-readable report
#   campaign_axis_audit.json   machine-readable roll-up
#   (stdout)                   short summary
#
# Correctness harness: smoke_test_campaign_axis_audit.py (same directory).
# =============================================================================

import argparse
import ast
import fnmatch
import json
import os
import sys
import time

import numpy as np


AUDIT_SCHEMA_VERSION = 1

# Fallback axis-group membership, used ONLY when --registry-src is not given
# or cannot be parsed.  Kept in sync with HPC_single_run.py by name.
FALLBACK_ASTRO_PARAMS = [
    'O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F', 'I_Theta', 'omega_I',
    'C_Theta', 'U_A', 'G_T', 'O_N',
]
FALLBACK_NEURON_PARAMS = [
    'Sigma', 'gbarA', 'delta_gA', 'tauA', 'VA', 'DeltaA', 'VR', 'I_inj', 'Cm',
    'DeltaT', 'VT', 'gL',
]
FALLBACK_SYNAPSE_PARAMS = [
    'EC50_ampa', 'EC50_nmda', 'U_0_ar', 'U_max', 'U_0_sr', 'Omega_f_sr',
    'Omega_f_ar', 'Omega_d', 'alpha_syn', 'g_ampa', 'g_nmda', 'x0', 'O_G',
    'Omega_G',
]

# Keys copied verbatim out of job_args.json into the per-campaign config
# signature.  A campaign whose tasks disagree on any of these is FLAGGED.
CONFIG_KEYS = [
    'mode', 'sweep_group', 'conn_rule', 'conn_periodic',
    'c_max', 'Nn', 'Na', 'density', 'density_astro',
    'simtime', 'n_topologies', 'n_params_per_worker',
    'topology_mode', 'displ_bias', 'gj_dist', 'gj_max_dist',
    'stoa_cutoff', 'stoa_sigma',
    'conn_prob_lo', 'conn_prob_hi',
    'p0_lo', 'p0_hi', 'd0_lo', 'd0_hi', 'beta_lo', 'beta_hi',
]


# =============================================================================
# 1. Discovery  (pure filesystem walking; no parsing)
# =============================================================================

def discover_campaigns(root, pattern='campaign_*', explicit=None):
    """Return sorted absolute paths of campaign directories.

    `explicit` (list of names or paths) overrides the glob when given.
    """
    root = os.path.abspath(root)
    if explicit:
        out = []
        for name in explicit:
            p = name if os.path.isabs(name) else os.path.join(root, name)
            out.append(os.path.abspath(p))
        return sorted(out)
    if not os.path.isdir(root):
        return []
    out = []
    with os.scandir(root) as it:
        for e in it:
            if e.is_dir() and fnmatch.fnmatch(e.name, pattern):
                out.append(os.path.abspath(e.path))
    return sorted(out)


def discover_tasks(campaign_dir):
    """Return sorted task directories inside a campaign.

    Two on-disk layouts are tolerated:
      nested : campaign_<TAG>/sweep_<NODETAG>_task<NNNN>/topo_*/
      flat   : campaign_<TAG>/topo_*/              (campaign IS the task)
    """
    if not os.path.isdir(campaign_dir):
        return []
    tasks = []
    with os.scandir(campaign_dir) as it:
        for e in it:
            if e.is_dir() and e.name.startswith('sweep_'):
                tasks.append(os.path.abspath(e.path))
    if tasks:
        return sorted(tasks)
    # flat layout: treat the campaign dir itself as a single task if it holds
    # either topo_* dirs or a job_args.json.
    has_topo = False
    with os.scandir(campaign_dir) as it:
        for e in it:
            if e.is_dir() and e.name.startswith('topo_'):
                has_topo = True
                break
    if has_topo or os.path.exists(os.path.join(campaign_dir, 'job_args.json')):
        return [os.path.abspath(campaign_dir)]
    return []


def discover_topos(task_dir, limit=None):
    """Return sorted topo_* directories inside one task directory."""
    if not os.path.isdir(task_dir):
        return []
    out = []
    with os.scandir(task_dir) as it:
        for e in it:
            if e.is_dir() and e.name.startswith('topo_'):
                out.append(os.path.abspath(e.path))
    out.sort()
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


def discover_iters(topo_dir, limit=None):
    """Return sorted iter_*.npz paths inside one topology directory."""
    if not os.path.isdir(topo_dir):
        return []
    out = []
    with os.scandir(topo_dir) as it:
        for e in it:
            if (e.is_file() and e.name.startswith('iter_')
                    and e.name.endswith('.npz')):
                out.append(os.path.abspath(e.path))
    out.sort()
    if limit is not None and limit > 0:
        out = out[:limit]
    return out


# =============================================================================
# 2. Metadata parsing  (JSON on disk -> plain dicts)
# =============================================================================

def read_json(path):
    """Read a small JSON file; return (obj, error_string_or_None)."""
    try:
        with open(path, 'r') as fh:
            return json.load(fh), None
    except Exception as exc:                                  # noqa: BLE001
        return None, '%s: %r' % (os.path.basename(path), exc)


def read_manifest_head(path, head_bytes=1 << 20, max_full_mb=200):
    """Parse manifest.json WITHOUT loading its 'topologies' array.

    rebuild_manifest() writes 'topologies' LAST (dict insertion order is
    preserved by json.dump), so everything the audit needs -- param_names,
    param_bounds, sweep_group, active_indices, axis_declaration, log_params --
    lives in the first few KB.  A full task manifest can be hundreds of MB;
    reading the head keeps this cheap on a networked filesystem.

    Returns (obj, error_string_or_None).  Falls back to a full parse only if
    the marker is absent AND the file is under `max_full_mb`.
    """
    marker = '"topologies":'
    try:
        with open(path, 'r') as fh:
            head = fh.read(head_bytes)
    except Exception as exc:                                  # noqa: BLE001
        return None, 'manifest read failed: %r' % (exc,)

    cut = head.find(marker)
    if cut >= 0:
        # Close the object after dropping the trailing comma of the previous
        # key/value pair, then re-attach 'topologies' as null so the shape of
        # the dict is unchanged for consumers.
        trimmed = head[:cut].rstrip()
        if trimmed.endswith(','):
            trimmed = trimmed[:-1]
        candidate = trimmed + ', "topologies": null}'
        try:
            return json.loads(candidate), None
        except Exception as exc:                              # noqa: BLE001
            return None, 'manifest head parse failed: %r' % (exc,)

    # Marker not in the head: either a tiny manifest fully contained in `head`
    # (then a direct parse works) or an unexpected key order.
    try:
        return json.loads(head), None
    except Exception:
        pass
    try:
        size_mb = os.path.getsize(path) / 1e6
    except OSError as exc:
        return None, 'manifest stat failed: %r' % (exc,)
    if size_mb > max_full_mb:
        return None, ('manifest too large for full parse (%.1f MB > %d MB) and '
                      'no "topologies" marker in the first %d bytes'
                      % (size_mb, max_full_mb, head_bytes))
    return read_json(path)


def read_task_metadata(task_dir, head_bytes, max_full_mb):
    """Collect job_args.json (+ manifest head) for one task directory."""
    rec = {'task_dir': task_dir, 'errors': []}

    ja, err = read_json(os.path.join(task_dir, 'job_args.json'))
    if err:
        rec['errors'].append(err)
    rec['job_args'] = ja

    mf, err = read_manifest_head(os.path.join(task_dir, 'manifest.json'),
                                 head_bytes=head_bytes, max_full_mb=max_full_mb)
    if err:
        rec['errors'].append(err)
    rec['manifest'] = mf
    return rec


# =============================================================================
# 3. Source-registry parsing  (ast only; the modules are NEVER imported)
# =============================================================================

def _eval_literalish(node):
    """ast node -> Python literal, tolerating np.array([...], dtype=...)."""
    try:
        return ast.literal_eval(node)
    except Exception:                                          # noqa: BLE001
        pass
    if isinstance(node, ast.Call) and node.args:
        try:
            return ast.literal_eval(node.args[0])
        except Exception:                                      # noqa: BLE001
            return None
    return None


def literal_from_source(path, name):
    """Return the value of top-level assignment `name` in `path`, or None.

    If the name is assigned several times at module level (e.g.
    'PARAM_NAMES = list(PARAM_NAMES)' followed by real content), the LAST
    successfully evaluated literal wins.  The module is parsed, never
    executed, so this is safe on files that import Brian2.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as fh:
            tree = ast.parse(fh.read())
    except Exception:                                          # noqa: BLE001
        return None
    found = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id == name:
                val = _eval_literalish(node.value)
                if val is not None:
                    found = val
    return found


def load_reference_registry(registry_src, sweep_src):
    """Best-effort read of the CURRENT on-disk registry, for comparison.

    Returns a dict with whatever could be parsed; missing pieces fall back to
    the module-level FALLBACK_* lists and are flagged in 'provenance'.
    """
    ref = {'provenance': {}}

    names = astro = neuron = synapse = nominal = None
    if registry_src and os.path.exists(registry_src):
        names = literal_from_source(registry_src, 'PARAM_NAMES')
        astro = literal_from_source(registry_src, 'ASTRO_PARAMS')
        neuron = literal_from_source(registry_src, 'NEURON_PARAMS')
        synapse = literal_from_source(registry_src, 'SYNAPSE_PARAMS')
        nominal = literal_from_source(registry_src, 'NOMINAL_PARAMS')

    ref['param_names'] = list(names) if names else None
    ref['provenance']['param_names'] = (
        registry_src if names else 'not available')

    ref['astro_params'] = list(astro) if astro else list(FALLBACK_ASTRO_PARAMS)
    ref['provenance']['astro_params'] = (
        registry_src if astro else 'built-in fallback list')

    ref['neuron_params'] = (list(neuron) if neuron
                            else list(FALLBACK_NEURON_PARAMS))
    ref['provenance']['neuron_params'] = (
        registry_src if neuron else 'built-in fallback list')

    ref['synapse_params'] = (list(synapse) if synapse
                             else list(FALLBACK_SYNAPSE_PARAMS))
    ref['provenance']['synapse_params'] = (
        registry_src if synapse else 'built-in fallback list')

    ref['nominal_params'] = [float(v) for v in nominal] if nominal else None
    ref['provenance']['nominal_params'] = (
        registry_src if nominal else 'not available')

    bounds = None
    if sweep_src and os.path.exists(sweep_src):
        bounds = literal_from_source(sweep_src, 'PARAM_BOUNDS')
    ref['current_param_bounds'] = (
        [[float(a), float(b)] for a, b in bounds] if bounds else None)
    ref['provenance']['current_param_bounds'] = (
        sweep_src if bounds else 'not available')

    kbounds = None
    if sweep_src and os.path.exists(sweep_src):
        kbounds = literal_from_source(sweep_src, 'KERNEL_BOUNDS')
    ref['current_kernel_bounds'] = (
        [[float(a), float(b)] for a, b in kbounds] if kbounds else None)
    ref['provenance']['current_kernel_bounds'] = (
        sweep_src if kbounds else 'not available')

    return ref


# =============================================================================
# 4. Empirical scan  (sample iter_*.npz; verify what actually varied)
# =============================================================================

def sample_iter_paths(task_dirs, n_target, rng, topos_per_task=3,
                      iters_per_topo=4):
    """Spread a sample of iter_*.npz paths across tasks and topologies.

    Sampling breadth matters more than depth here: an axis frozen in one task
    but swept in another is exactly the kind of drift this audit must catch,
    so every task contributes before any task contributes twice.
    """
    if n_target <= 0 or not task_dirs:
        return []
    picked = []
    order = list(task_dirs)
    rng.shuffle(order)
    for td in order:
        topos = discover_topos(td)
        if not topos:
            continue
        rng.shuffle(topos)
        for tp in topos[:max(1, topos_per_task)]:
            its = discover_iters(tp)
            if not its:
                continue
            rng.shuffle(its)
            picked.extend(its[:max(1, iters_per_topo)])
            if len(picked) >= n_target:
                return picked[:n_target]
    return picked[:n_target]


def scan_iter_files(paths, fallback_names=None):
    """Read 'params' (+ connectivity keys) from a sample of iter_*.npz.

    Returns dict:
      n_read           : int
      n_failed         : int
      errors           : list[str] (first few)
      param_names      : list[str] or None (from the iter_*.json sidecar)
      per_axis         : {name: {'n_distinct', 'min', 'max', 'constant'}}
      conn             : {key: {'n_distinct','min','max'}}  (conn_prob or
                         p0_conn/d0_conn/beta_conn, whichever the rule wrote)
      conn_rules       : sorted list of distinct 'conn_rule' strings seen
      noise_modes      : sorted list of distinct 'noise_mode' strings seen
    """
    rows = []
    conn_cols = {}
    conn_rules = set()
    noise_modes = set()
    errors = []
    n_failed = 0
    names = None

    for p in paths:
        try:
            with np.load(p, allow_pickle=False) as z:
                if 'params' not in z.files:
                    raise KeyError("no 'params' array in npz")
                rows.append(np.asarray(z['params'], dtype=np.float64))
                for key in ('conn_prob', 'p0_conn', 'd0_conn', 'beta_conn'):
                    if key in z.files:
                        conn_cols.setdefault(key, []).append(float(z[key]))
                if 'conn_rule' in z.files:
                    conn_rules.add(str(z['conn_rule']))
                if 'noise_mode' in z.files:
                    noise_modes.add(str(z['noise_mode']))
        except Exception as exc:                              # noqa: BLE001
            n_failed += 1
            if len(errors) < 5:
                errors.append('%s: %r' % (os.path.basename(p), exc))
            continue
        if names is None:
            side = p[:-4] + '.json'
            sc, _ = read_json(side)
            if isinstance(sc, dict) and isinstance(sc.get('param_names'), list):
                names = list(sc['param_names'])

    out = {'n_read': len(rows), 'n_failed': n_failed, 'errors': errors,
           'param_names': names, 'per_axis': {}, 'conn': {},
           'conn_rules': sorted(conn_rules),
           'noise_modes': sorted(noise_modes)}
    if not rows:
        return out

    widths = sorted({r.shape[0] for r in rows})
    out['param_vector_widths'] = [int(w) for w in widths]
    width = widths[-1]
    mat = np.full((len(rows), width), np.nan, dtype=np.float64)
    for k, r in enumerate(rows):
        mat[k, :r.shape[0]] = r

    if names is None:
        names = list(fallback_names) if fallback_names else []
    if len(names) < width:
        names = list(names) + ['axis_%d' % j for j in range(len(names), width)]

    for j in range(width):
        col = mat[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        uniq = np.unique(col)
        out['per_axis'][names[j]] = {
            'index': int(j),
            'n_distinct': int(uniq.size),
            'min': float(col.min()),
            'max': float(col.max()),
            'constant': bool(uniq.size == 1),
        }

    for key, vals in conn_cols.items():
        a = np.asarray(vals, dtype=np.float64)
        out['conn'][key] = {'n_distinct': int(np.unique(a).size),
                            'min': float(a.min()), 'max': float(a.max()),
                            'n': int(a.size)}
    return out


def scan_topology_meta(task_dirs, rng, n_target=40):
    """Sample topology_meta.json to recover per-topology geometry + kernel."""
    picked = []
    order = list(task_dirs)
    rng.shuffle(order)
    for td in order:
        for tp in discover_topos(td, limit=None)[:5]:
            picked.append(os.path.join(tp, 'topology_meta.json'))
            if len(picked) >= n_target:
                break
        if len(picked) >= n_target:
            break

    acc = {}
    n_read = 0
    for p in picked:
        obj, _ = read_json(p)
        if not isinstance(obj, dict):
            continue
        n_read += 1
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc.setdefault(k, []).append(float(v))
    stats = {}
    for k, vals in acc.items():
        a = np.asarray(vals, dtype=np.float64)
        stats[k] = {'min': float(a.min()), 'max': float(a.max()),
                    'mean': float(a.mean()), 'n': int(a.size)}
    return {'n_read': n_read, 'stats': stats}


# =============================================================================
# 5. Per-campaign aggregation
# =============================================================================

def _cfg_value(job_args, key):
    if not isinstance(job_args, dict):
        return None
    if key == 'sweep_group':
        return job_args.get('sweep_group', job_args.get('_sweep_group'))
    return job_args.get(key)


def aggregate_campaign(campaign_dir, args, rng, reference):
    """Build the full audit record for one campaign directory."""
    tasks = discover_tasks(campaign_dir)
    rec = {
        'campaign': os.path.basename(campaign_dir.rstrip(os.sep)),
        'campaign_dir': campaign_dir,
        'n_task_dirs': len(tasks),
        'warnings': [],
        'errors': [],
    }
    if not tasks:
        rec['warnings'].append('no sweep_*/ task directories and no topo_*/ '
                               'found -- campaign is empty or not yet started')
        return rec

    scan_tasks = tasks if args.max_tasks <= 0 else tasks[:args.max_tasks]
    if len(scan_tasks) < len(tasks):
        rec['warnings'].append(
            'metadata read limited to the first %d of %d task dirs by '
            '--max-tasks' % (len(scan_tasks), len(tasks)))

    metas = [read_task_metadata(td, args.manifest_head_bytes,
                                args.max_manifest_mb) for td in scan_tasks]
    for m in metas:
        rec['errors'].extend(m['errors'])

    job_args_list = [m['job_args'] for m in metas
                     if isinstance(m['job_args'], dict)]
    manifests = [m['manifest'] for m in metas if isinstance(m['manifest'], dict)]
    rec['n_job_args_read'] = len(job_args_list)
    rec['n_manifests_read'] = len(manifests)

    # ---- config signature, and any disagreement between tasks --------------
    cfg = {}
    cfg_variants = {}
    for key in CONFIG_KEYS:
        vals = []
        for ja in job_args_list:
            v = _cfg_value(ja, key)
            if v is not None and v not in vals:
                vals.append(v)
        if len(vals) == 1:
            cfg[key] = vals[0]
        elif len(vals) > 1:
            cfg[key] = vals[0]
            cfg_variants[key] = vals
    rec['config'] = cfg
    if cfg_variants:
        rec['config_disagreements'] = cfg_variants
        rec['warnings'].append(
            'tasks within this campaign disagree on: %s -- the campaign is NOT '
            'a single homogeneous configuration'
            % ', '.join(sorted(cfg_variants)))

    # derived geometry ------------------------------------------------------
    c_max = cfg.get('c_max')
    Nn = cfg.get('Nn')
    Na = cfg.get('Na')
    geom = {'c_max_um': c_max, 'Nn': Nn, 'Na': Na,
            'density_arg': cfg.get('density'),
            'density_astro_arg': cfg.get('density_astro'),
            'topology_mode': cfg.get('topology_mode')}
    if c_max:
        area_mm2 = (float(c_max) / 1000.0) ** 2
        geom['area_mm2'] = area_mm2
        geom['side_um'] = float(c_max)
        if Nn:
            geom['rho_neuron_per_mm2'] = float(Nn) / area_mm2
        if Na:
            geom['rho_astro_per_mm2'] = float(Na) / area_mm2
    rec['geometry'] = geom

    # ---- declared axes -----------------------------------------------------
    decl = {}
    src_ja = None
    for ja in job_args_list:
        if isinstance(ja.get('_axis_declaration'), dict):
            src_ja = ja
            break
    if src_ja is not None:
        ad = src_ja['_axis_declaration']
        decl['source'] = 'job_args.json:_axis_declaration'
        decl['swept'] = sorted(ad.get('swept_axes', {}).keys())
        decl['swept_detail'] = ad.get('swept_axes', {})
        decl['consumed'] = sorted(ad.get('consumed_axes', []))
        decl['fixed'] = ad.get('fixed_axes', {})
        decl['inert'] = ad.get('inert_axes', {})
    else:
        mf = manifests[0] if manifests else None
        if isinstance(mf, dict) and isinstance(mf.get('axis_declaration'), dict):
            ad = mf['axis_declaration']
            decl['source'] = 'manifest.json:axis_declaration'
            decl['swept'] = sorted(ad.get('swept_axes', {}).keys())
            decl['swept_detail'] = ad.get('swept_axes', {})
            decl['consumed'] = sorted(ad.get('consumed_axes', []))
            decl['fixed'] = ad.get('fixed_axes', {})
            decl['inert'] = ad.get('inert_axes', {})
        else:
            names = None
            for ja in job_args_list:
                if isinstance(ja.get('_active_param_names'), list):
                    names = list(ja['_active_param_names'])
                    break
            if names is None:
                for mf2 in manifests:
                    if isinstance(mf2.get('active_param_names'), list):
                        names = list(mf2['active_param_names'])
                        break
            if names is not None:
                decl['source'] = ('_active_param_names (pre-axis_declaration '
                                  'campaign: no swept/consumed/inert split '
                                  'was recorded)')
                decl['swept'] = sorted(names)
                decl['swept_detail'] = {}
                decl['consumed'] = []
                decl['fixed'] = {}
                decl['inert'] = {}
                rec['warnings'].append(
                    'this campaign predates axis_declaration; '
                    'swept/consumed/inert cannot be distinguished from '
                    'metadata alone and is inferred from mode below')
            else:
                decl['source'] = 'NONE FOUND'
                decl['swept'] = []
                decl['swept_detail'] = {}
                decl['consumed'] = []
                decl['fixed'] = {}
                decl['inert'] = {}
                rec['warnings'].append(
                    'no axis declaration and no _active_param_names in any '
                    'job_args.json -- declared axis set is unknown; rely on '
                    'the empirical scan')
    rec['declared'] = decl

    # ---- registry + bounds actually recorded by this campaign --------------
    reg = {}
    mf = manifests[0] if manifests else None
    if isinstance(mf, dict):
        reg['manifest_version'] = mf.get('manifest_version')
        reg['n_dims'] = mf.get('n_dims')
        reg['param_names'] = mf.get('param_names')
        reg['param_units'] = mf.get('param_units')
        reg['param_bounds'] = mf.get('param_bounds')
        reg['log_params'] = mf.get('log_params')
        reg['log_transform'] = mf.get('log_transform')
        reg['sweep_group'] = mf.get('sweep_group')
        reg['active_param_names'] = mf.get('active_param_names')
    rec['registry'] = reg

    # bounds consistency across the tasks that were read ---------------------
    sigs = set()
    for mf2 in manifests:
        pb = mf2.get('param_bounds')
        if isinstance(pb, list):
            sigs.add(json.dumps(pb))
    if len(sigs) > 1:
        rec['warnings'].append(
            'param_bounds DIFFER between tasks of this campaign (%d distinct '
            'bound matrices) -- tasks were launched against different code '
            'versions' % len(sigs))
    rec['n_distinct_param_bounds'] = len(sigs)

    # ---- seeds -------------------------------------------------------------
    seeds = []
    seed_by_task = {}
    for m in metas:
        ja = m['job_args']
        if not isinstance(ja, dict):
            continue
        s = ja.get('_resolved_seed_master', ja.get('seed_master'))
        if s is None:
            continue
        try:
            s = int(s)
        except (TypeError, ValueError):
            continue
        seeds.append(s)
        seed_by_task[os.path.basename(m['task_dir'])] = s
    arr = np.asarray(sorted(seeds), dtype=np.int64)
    rec['seeds'] = {
        'n_tasks_with_seed': int(arr.size),
        'n_distinct': int(np.unique(arr).size) if arr.size else 0,
        'min': int(arr.min()) if arr.size else None,
        'max': int(arr.max()) if arr.size else None,
        'values': [int(v) for v in np.unique(arr)] if arr.size else [],
        'by_task': seed_by_task,
    }
    if arr.size and np.unique(arr).size < arr.size:
        rec['warnings'].append(
            'seed_master REUSED within this campaign: %d tasks, only %d '
            'distinct seeds (reuse factor %.2f) -- those tasks drew the SAME '
            'topologies and the SAME parameter vectors'
            % (arr.size, np.unique(arr).size, arr.size / np.unique(arr).size))

    # ---- empirical scan ----------------------------------------------------
    if args.quick:
        rec['empirical'] = {'skipped': True,
                            'reason': '--quick: no iter_*.npz were read'}
        rec['topology_meta'] = {'skipped': True}
    else:
        paths = sample_iter_paths(scan_tasks, args.iters_per_campaign, rng,
                                  topos_per_task=args.topos_per_task,
                                  iters_per_topo=args.iters_per_topo)
        fallback_names = reg.get('param_names') or reference.get('param_names')
        rec['empirical'] = scan_iter_files(paths, fallback_names=fallback_names)
        rec['empirical']['n_sampled_paths'] = len(paths)
        rec['topology_meta'] = scan_topology_meta(
            scan_tasks, rng, n_target=args.topo_meta_sample)

    return rec


# =============================================================================
# 6. Classification  (declared x empirical x consumption)
# =============================================================================

def classify_axes(rec, reference):
    """Per-axis status for one campaign.

    status is one of:
      'swept+consumed'  declared swept, simulator reads it, column varies
      'swept+inert'     declared swept, nothing reads it (mode/rule dependent)
      'swept-but-const' declared swept, but the sampled column never varied
      'varies-undeclared' column varies although the axis was not declared swept
      'fixed'           held at a constant value (value reported)

    An axis that does not exist in this campaign's registry is simply ABSENT
    from the returned mapping; the report renders it as '-'.
    """
    decl = rec.get('declared', {})
    emp = rec.get('empirical', {}) or {}
    per_axis = emp.get('per_axis', {}) or {}
    reg_names = (rec.get('registry', {}) or {}).get('param_names')
    if not reg_names:
        reg_names = emp.get('param_names') or reference.get('param_names') or []

    swept = set(decl.get('swept', []))
    consumed = set(decl.get('consumed', []))
    inert = set(decl.get('inert', {}) or {})
    fixed = dict(decl.get('fixed', {}) or {})

    mode = (rec.get('config', {}) or {}).get('mode')
    astro_names = set(reference.get('astro_params', FALLBACK_ASTRO_PARAMS))

    # Pre-axis_declaration campaigns record no consumed/inert split; infer it
    # from mode, which is the only thing that decides whether astrocyte axes
    # reach a network object at all.
    inferred = not consumed and swept
    if inferred and mode is not None and mode != 'Full':
        inert = set(inert) | (swept & astro_names)

    out = {}
    for name in reg_names:
        st = {'name': name}
        e = per_axis.get(name)
        st['n_distinct'] = e['n_distinct'] if e else None
        st['observed_min'] = e['min'] if e else None
        st['observed_max'] = e['max'] if e else None

        in_swept = name in swept
        is_inert = name in inert
        varies = bool(e and not e['constant'])

        if in_swept and is_inert:
            st['status'] = 'swept+inert'
        elif in_swept and e is not None and not varies:
            st['status'] = 'swept-but-const'
        elif in_swept:
            st['status'] = 'swept+consumed'
        elif varies:
            st['status'] = 'varies-undeclared'
        else:
            st['status'] = 'fixed'

        if st['status'] == 'fixed':
            if name in fixed:
                st['fixed_value'] = fixed[name]
            elif e is not None and e['constant']:
                st['fixed_value'] = e['min']
            else:
                st['fixed_value'] = None
        if in_swept:
            det = (decl.get('swept_detail', {}) or {}).get(name)
            if isinstance(det, dict):
                st['declared_low'] = det.get('low')
                st['declared_high'] = det.get('high')
                st['declared_log'] = det.get('log')
                st['declared_level'] = det.get('level')
            else:
                # Pre-axis_declaration campaign: no per-axis swept_detail was
                # written, but manifest.json still carries the full
                # param_bounds matrix. Backfill from there so the bounds a
                # campaign actually sampled under are never missing from the
                # report just because the schema was older.
                reg = rec.get('registry', {}) or {}
                pn, pb = reg.get('param_names'), reg.get('param_bounds')
                if (isinstance(pn, list) and isinstance(pb, list)
                        and name in pn and pn.index(name) < len(pb)):
                    lo, hi = pb[pn.index(name)]
                    st['declared_low'] = float(lo)
                    st['declared_high'] = float(hi)
                    st['declared_level'] = 'run_args'
                    st['bounds_source'] = 'manifest.json:param_bounds'
        st['is_astro_axis'] = name in astro_names
        st['is_consumed'] = (name in consumed) if consumed else (not is_inert)
        out[name] = st
    rec['axis_status'] = out
    rec['inference_note'] = ('consumed/inert inferred from mode (no '
                             'axis_declaration on disk)' if inferred else
                             'consumed/inert read from axis_declaration')
    return out


def cross_campaign_seed_overlap(records):
    """Pairwise intersection sizes of the per-campaign seed_master sets."""
    sets = {}
    for r in records:
        vals = (r.get('seeds', {}) or {}).get('values', [])
        sets[r['campaign']] = set(int(v) for v in vals)
    names = sorted(sets)
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            inter = sets[na] & sets[nb]
            pairs.append({
                'a': na, 'b': nb,
                'n_a': len(sets[na]), 'n_b': len(sets[nb]),
                'n_overlap': len(inter),
                'overlap_examples': sorted(inter)[:10],
                'prefix_containment': bool(
                    inter and (inter == sets[na] or inter == sets[nb])),
            })
    return {'campaigns': names, 'pairs': pairs}


# =============================================================================
# 7. Reporting
# =============================================================================

def _fmt(v, nd=4):
    if v is None:
        return '-'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if isinstance(v, float):
        if v != v:
            return 'nan'
        if v == 0:
            return '0'
        av = abs(v)
        if av >= 1e4 or av < 1e-3:
            return '%.3e' % v
        return ('%.' + str(nd) + 'g') % v
    return str(v)


def _cell_for_axis(st):
    s = st.get('status')
    if s == 'absent':
        return '-'
    if s in ('swept+consumed', 'swept+inert', 'swept-but-const'):
        lo, hi = st.get('declared_low'), st.get('declared_high')
        rng = ('[%s, %s]' % (_fmt(lo), _fmt(hi))) if lo is not None else ''
        tag = {'swept+consumed': 'SWEPT', 'swept+inert': 'SWEPT(INERT)',
               'swept-but-const': 'SWEPT(const!)'}[s]
        return ('%s %s' % (tag, rng)).strip()
    if s == 'varies-undeclared':
        return 'VARIES(undeclared) [%s, %s]' % (_fmt(st.get('observed_min')),
                                                _fmt(st.get('observed_max')))
    return 'fixed=%s' % _fmt(st.get('fixed_value'))


def render_markdown(records, reference, overlap, args):
    L = []
    A = L.append
    A('# Campaign axis / seed / geometry audit')
    A('')
    A('- Generated: %s' % time.strftime('%Y-%m-%dT%H:%M:%S'))
    A('- Root: `%s`' % os.path.abspath(args.root))
    A('- Campaign glob: `%s`' % args.glob)
    A('- Audit schema version: %d' % AUDIT_SCHEMA_VERSION)
    A('- Mode: %s' % ('QUICK (metadata only; no iter_*.npz read)'
                      if args.quick else
                      'full (metadata + empirical iter_*.npz scan)'))
    A('')
    A('Every statement below is derived from files on disk. Nothing was '
      'imported or executed; registry files were parsed with `ast`.')
    A('')

    # ---- 1. inventory -----------------------------------------------------
    A('## 1. Campaign inventory')
    A('')
    hdr = ('| campaign | tasks | job_args | mode | sweep_group | conn_rule | '
           'periodic | c_max [um] | Nn | Na | simtime [s] | n_topo | k | '
           'registry n_dims | manifest v |')
    A(hdr)
    A('|' + '---|' * 15)
    for r in records:
        c = r.get('config', {}) or {}
        reg = r.get('registry', {}) or {}
        A('| %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | '
          '%s | %s |' % (
              r['campaign'], r.get('n_task_dirs', 0),
              r.get('n_job_args_read', 0),
              _fmt(c.get('mode')), _fmt(c.get('sweep_group')),
              _fmt(c.get('conn_rule')), _fmt(c.get('conn_periodic')),
              _fmt(c.get('c_max')), _fmt(c.get('Nn')), _fmt(c.get('Na')),
              _fmt(c.get('simtime')), _fmt(c.get('n_topologies')),
              _fmt(c.get('n_params_per_worker')),
              _fmt(reg.get('n_dims')), _fmt(reg.get('manifest_version'))))
    A('')

    # ---- 2. geometry ------------------------------------------------------
    A('## 2. Culture geometry actually used')
    A('')
    A('The culture is a 2-D square arena `[0, c_max]^2` (um); neuron and '
      'astrocyte somata are drawn uniformly inside it. Densities below are '
      'BACK-COMPUTED from the recorded Nn / Na and c_max, so they are correct '
      'whether or not `--density` was passed.')
    A('')
    A('| campaign | side [um] | area [mm^2] | Nn | Na | rho_neuron [1/mm^2] | '
      'rho_astro [1/mm^2] | --density arg | --density_astro arg | topology_mode |')
    A('|' + '---|' * 10)
    for r in records:
        g = r.get('geometry', {}) or {}
        A('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['campaign'], _fmt(g.get('side_um')), _fmt(g.get('area_mm2')),
            _fmt(g.get('Nn')), _fmt(g.get('Na')),
            _fmt(g.get('rho_neuron_per_mm2'), 6),
            _fmt(g.get('rho_astro_per_mm2'), 6),
            _fmt(g.get('density_arg')), _fmt(g.get('density_astro_arg')),
            _fmt(g.get('topology_mode'))))
    A('')

    # ---- 3. axis matrix ---------------------------------------------------
    A('## 3. Swept-axis matrix (aligned BY NAME, not by index)')
    A('')
    A('Registry dimensionality changed between campaigns, so index alignment '
      'is unsafe; every axis below is keyed by name. Legend: `SWEPT [lo, hi]` '
      '= declared swept and read by the simulator; `SWEPT(INERT)` = drawn and '
      'recorded but read by nothing (no astrocyte object is instantiated '
      'unless mode=Full); `SWEPT(const!)` = declared swept but the sampled '
      'column never varied; `VARIES(undeclared)` = column varies although the '
      'declaration does not list it; `fixed=V` = held at V; `-` = the axis '
      'does not exist in that campaign\'s registry.')
    A('')
    all_names = []
    for r in records:
        for n in (r.get('registry', {}) or {}).get('param_names') or []:
            if n not in all_names:
                all_names.append(n)
    for r in records:
        for n in (r.get('axis_status', {}) or {}):
            if n not in all_names:
                all_names.append(n)
    A('| axis | astro? | ' + ' | '.join(r['campaign'] for r in records) + ' |')
    A('|' + '---|' * (2 + len(records)))
    for name in all_names:
        cells = []
        is_astro = False
        for r in records:
            st = (r.get('axis_status', {}) or {}).get(name)
            if st is None:
                cells.append('-')
            else:
                is_astro = is_astro or bool(st.get('is_astro_axis'))
                cells.append(_cell_for_axis(st))
        A('| `%s` | %s | %s |' % (name, 'Y' if is_astro else '',
                                  ' | '.join(cells)))
    A('')

    # ---- 4. astrocyte verdict ---------------------------------------------
    A('## 4. Astrocyte axes: swept, and were they read?')
    A('')
    A('| campaign | mode | astro axes swept | of which INERT | verdict |')
    A('|' + '---|' * 5)
    for r in records:
        stat = r.get('axis_status', {}) or {}
        astro = [n for n, s in stat.items() if s.get('is_astro_axis')]
        sw = [n for n in astro
              if stat[n]['status'].startswith('swept')]
        inert = [n for n in sw if stat[n]['status'] == 'swept+inert']
        mode = (r.get('config', {}) or {}).get('mode')
        if not sw:
            verdict = 'NO astrocyte axis was swept'
        elif inert:
            verdict = ('astro axes drawn but NOT consumed (mode=%s builds no '
                       'astrocyte objects)' % _fmt(mode))
        else:
            verdict = 'astro axes swept AND consumed (mode=Full)'
        A('| %s | %s | %d | %d | %s |'
          % (r['campaign'], _fmt(mode), len(sw), len(inert), verdict))
        if sw:
            A('| | | `%s` | | |' % ', '.join(sorted(sw)))
    A('')

    # ---- 5. empirical verification ----------------------------------------
    A('## 5. Empirical verification (sampled iter_*.npz)')
    A('')
    if args.quick:
        A('Skipped (`--quick`). The declared columns above were NOT '
          'cross-checked against what the sampler actually wrote.')
    else:
        A('| campaign | npz read | failed | vector width(s) | axes varying | '
          'axes constant | conn_rule in npz |')
        A('|' + '---|' * 7)
        for r in records:
            e = r.get('empirical', {}) or {}
            pa = e.get('per_axis', {}) or {}
            nvary = sum(1 for v in pa.values() if not v['constant'])
            nconst = sum(1 for v in pa.values() if v['constant'])
            A('| %s | %s | %s | %s | %d | %d | %s |' % (
                r['campaign'], _fmt(e.get('n_read')), _fmt(e.get('n_failed')),
                ','.join(str(w) for w in e.get('param_vector_widths', [])) or '-',
                nvary, nconst, ','.join(e.get('conn_rules', [])) or '-'))
        A('')
        A('### 5.1 Disagreements between declaration and data')
        A('')
        any_bad = False
        for r in records:
            bad = [(n, s) for n, s in (r.get('axis_status', {}) or {}).items()
                   if s['status'] in ('swept-but-const', 'varies-undeclared')]
            if bad:
                any_bad = True
                A('**%s**' % r['campaign'])
                for n, s in sorted(bad):
                    A('- `%s`: %s (observed %s distinct value(s) in [%s, %s])'
                      % (n, s['status'], _fmt(s.get('n_distinct')),
                         _fmt(s.get('observed_min')), _fmt(s.get('observed_max'))))
                A('')
        if not any_bad:
            A('None. Every declared-swept axis varied in the sample and no '
              'undeclared axis varied.')
        A('')

    # ---- 6. seeds ---------------------------------------------------------
    A('## 6. Seed footprint and cross-campaign collisions')
    A('')
    A('`SEED_MASTER` fixes a task\'s entire scientific content: topology '
      'draws, kernel draws, every theta, every per-run seed. Two tasks with '
      'the same seed_master and the same code produce the SAME simulations.')
    A('')
    A('| campaign | tasks with seed | distinct seeds | min | max | reuse factor |')
    A('|' + '---|' * 6)
    for r in records:
        s = r.get('seeds', {}) or {}
        n, d = s.get('n_tasks_with_seed', 0), s.get('n_distinct', 0)
        A('| %s | %d | %d | %s | %s | %s |'
          % (r['campaign'], n, d, _fmt(s.get('min')), _fmt(s.get('max')),
             ('%.2f' % (n / d)) if d else '-'))
    A('')
    A('### 6.1 Pairwise seed-set overlap')
    A('')
    A('| campaign A | campaign B | n seeds A | n seeds B | overlap | '
      'containment |')
    A('|' + '---|' * 6)
    for p in overlap.get('pairs', []):
        A('| %s | %s | %d | %d | %d | %s |'
          % (p['a'], p['b'], p['n_a'], p['n_b'], p['n_overlap'],
             'YES' if p['prefix_containment'] else
             ('partial' if p['n_overlap'] else 'none')))
    A('')
    A('A non-zero overlap means the two campaigns contain literally duplicated '
      'simulations wherever the code was also unchanged; "containment" means '
      'one campaign\'s seed set is a subset of the other\'s.')
    A('')

    # ---- 7. connectivity prior --------------------------------------------
    A('## 7. Connectivity prior actually drawn')
    A('')
    A('| campaign | conn_rule | periodic | observed ranges |')
    A('|' + '---|' * 4)
    for r in records:
        c = r.get('config', {}) or {}
        e = r.get('empirical', {}) or {}
        conn = e.get('conn', {}) or {}
        parts = []
        for k in ('conn_prob', 'p0_conn', 'd0_conn', 'beta_conn'):
            if k in conn:
                parts.append('%s in [%s, %s] (%d distinct)'
                             % (k, _fmt(conn[k]['min']), _fmt(conn[k]['max']),
                                conn[k]['n_distinct']))
        A('| %s | %s | %s | %s |' % (r['campaign'], _fmt(c.get('conn_rule')),
                                     _fmt(c.get('conn_periodic')),
                                     '; '.join(parts) if parts else '-'))
    A('')

    # ---- 8. reference comparison ------------------------------------------
    A('## 8. Comparison against the CURRENT on-disk registry')
    A('')
    cur = reference.get('current_param_bounds')
    names = reference.get('param_names')
    if cur and names:
        A('Source: `%s` (PARAM_BOUNDS) and `%s` (PARAM_NAMES), parsed with '
          '`ast`.' % (reference['provenance']['current_param_bounds'],
                      reference['provenance']['param_names']))
        A('')
        A('| axis | current [lo, hi] | ' +
          ' | '.join(r['campaign'] for r in records) + ' |')
        A('|' + '---|' * (2 + len(records)))
        for j, nm in enumerate(names):
            if j >= len(cur):
                break
            row = ['[%s, %s]' % (_fmt(cur[j][0]), _fmt(cur[j][1]))]
            differs = False
            for r in records:
                reg = r.get('registry', {}) or {}
                pn = reg.get('param_names') or []
                pb = reg.get('param_bounds') or []
                if nm in pn and pn.index(nm) < len(pb):
                    lo, hi = pb[pn.index(nm)]
                    row.append('[%s, %s]' % (_fmt(lo), _fmt(hi)))
                    if (abs(float(lo) - float(cur[j][0])) > 1e-12
                            or abs(float(hi) - float(cur[j][1])) > 1e-12):
                        differs = True
                else:
                    row.append('-')
                    differs = True
            mark = ' **<-- differs**' if differs else ''
            A('| `%s` | %s |%s' % (nm, ' | '.join(row), mark))
        A('')
    else:
        A('Not available: pass `--registry-src <HPC_single_run.py>` and '
          '`--sweep-src <HPC_main_sweep.py>` to enable this comparison.')
        A('')

    # ---- 9. warnings ------------------------------------------------------
    A('## 9. Warnings and read errors')
    A('')
    any_w = False
    for r in records:
        ws = list(r.get('warnings', [])) + list(r.get('errors', []))[:10]
        if ws:
            any_w = True
            A('**%s**' % r['campaign'])
            for w in ws:
                A('- %s' % w)
            A('')
    if not any_w:
        A('None.')
        A('')
    return '\n'.join(L) + '\n'


def print_summary(records, overlap):
    print('')
    print('=' * 78)
    print('CAMPAIGN AXIS AUDIT -- summary')
    print('=' * 78)
    for r in records:
        c = r.get('config', {}) or {}
        g = r.get('geometry', {}) or {}
        stat = r.get('axis_status', {}) or {}
        sw = [n for n, s in stat.items() if s['status'].startswith('swept')]
        astro_sw = [n for n in sw if stat[n].get('is_astro_axis')]
        inert = [n for n in sw if stat[n]['status'] == 'swept+inert']
        s = r.get('seeds', {}) or {}
        print('')
        print('%s' % r['campaign'])
        print('  tasks=%d  mode=%s  sweep_group=%s  conn_rule=%s'
              % (r.get('n_task_dirs', 0), c.get('mode'), c.get('sweep_group'),
                 c.get('conn_rule')))
        print('  culture: %s um square  Nn=%s  Na=%s  rho_n=%s /mm^2'
              % (_fmt(g.get('side_um')), _fmt(g.get('Nn')), _fmt(g.get('Na')),
                 _fmt(g.get('rho_neuron_per_mm2'))))
        print('  swept axes: %d  (astro %d, of which inert %d)'
              % (len(sw), len(astro_sw), len(inert)))
        print('  seeds: %d task(s), %d distinct, range [%s, %s]'
              % (s.get('n_tasks_with_seed', 0), s.get('n_distinct', 0),
                 _fmt(s.get('min')), _fmt(s.get('max'))))
        for w in r.get('warnings', []):
            print('  WARNING: %s' % w)
    print('')
    print('Pairwise seed overlap:')
    for p in overlap.get('pairs', []):
        flag = ''
        if p['n_overlap']:
            flag = '  <-- COLLISION'
        if p['prefix_containment']:
            flag = '  <-- CONTAINMENT'
        print('  %-34s vs %-34s : %d%s'
              % (p['a'], p['b'], p['n_overlap'], flag))
    print('')


# =============================================================================
# 8. CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=('Read-only audit of which parameters were swept, which '
                     'seeds were used, and what culture geometry was '
                     'simulated, across a cohort of campaign_* directories.'))
    p.add_argument('root', nargs='?', default='.',
                   help='directory holding the campaign_* dirs (default: .)')
    p.add_argument('--glob', default='campaign_*',
                   help='campaign directory pattern (default: %(default)s)')
    p.add_argument('--campaigns', nargs='+', default=None,
                   help='explicit campaign dir names/paths; overrides --glob')
    p.add_argument('--out', default='./campaign_axis_audit',
                   help='output directory (created; default: %(default)s)')
    p.add_argument('--quick', action='store_true',
                   help='metadata only: do not open any iter_*.npz')
    p.add_argument('--iters-per-campaign', type=int, default=300,
                   help='iter_*.npz sampled per campaign (default: %(default)s)')
    p.add_argument('--topos-per-task', type=int, default=3,
                   help='topologies sampled per task (default: %(default)s)')
    p.add_argument('--iters-per-topo', type=int, default=4,
                   help='iters sampled per topology (default: %(default)s)')
    p.add_argument('--topo-meta-sample', type=int, default=40,
                   help='topology_meta.json files sampled (default: %(default)s)')
    p.add_argument('--max-tasks', type=int, default=0,
                   help='cap task dirs read per campaign (0 = all)')
    p.add_argument('--manifest-head-bytes', type=int, default=1 << 20,
                   help='bytes of manifest.json read before the topologies '
                        'array (default: 1048576)')
    p.add_argument('--max-manifest-mb', type=int, default=200,
                   help='refuse a full manifest parse above this size')
    p.add_argument('--registry-src', default=None,
                   help='path to HPC_single_run.py (PARAM_NAMES, ASTRO_PARAMS, '
                        'NOMINAL_PARAMS) -- parsed with ast, never imported')
    p.add_argument('--sweep-src', default=None,
                   help='path to HPC_main_sweep.py (PARAM_BOUNDS, '
                        'KERNEL_BOUNDS) -- parsed with ast, never imported')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for file sampling (default: %(default)s)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    campaigns = discover_campaigns(args.root, args.glob, args.campaigns)
    if not campaigns:
        print('[audit] no campaign directories matched %r under %s'
              % (args.glob, os.path.abspath(args.root)), file=sys.stderr)
        return 2

    reference = load_reference_registry(args.registry_src, args.sweep_src)
    rng = np.random.default_rng(args.seed)

    records = []
    for cdir in campaigns:
        print('[audit] scanning %s ...' % os.path.basename(cdir), flush=True)
        rec = aggregate_campaign(cdir, args, rng, reference)
        classify_axes(rec, reference)
        records.append(rec)

    overlap = cross_campaign_seed_overlap(records)

    os.makedirs(args.out, exist_ok=True)
    md = render_markdown(records, reference, overlap, args)
    md_path = os.path.join(args.out, 'campaign_axis_audit.md')
    with open(md_path, 'w') as fh:
        fh.write(md)

    payload = {
        'audit_schema_version': AUDIT_SCHEMA_VERSION,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'root': os.path.abspath(args.root),
        'glob': args.glob,
        'quick': bool(args.quick),
        'reference': reference,
        'campaigns': records,
        'seed_overlap': overlap,
    }
    js_path = os.path.join(args.out, 'campaign_axis_audit.json')
    with open(js_path, 'w') as fh:
        json.dump(payload, fh, indent=2, default=str)

    print_summary(records, overlap)
    print('[audit] wrote %s' % md_path)
    print('[audit] wrote %s' % js_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
