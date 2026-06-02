import os
import gc
import time
import random
import warnings
import json
import copy
import hashlib
import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler, FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE, ADASYN

try:
    import cupy as cp
except ImportError:
    cp = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")
warnings.filterwarnings("ignore", message=".*Found Intel OpenMP.*")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Series): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Trainer:
    """
    Framework Unificado de Benchmarking EEG.
    - Grid Search con checkpoint robusto (fold-indexado).
    - Métodos estocásticos con cache atómico y control de RAM.
    - Submuestreo estratificado con control estricto de varianza (<2%).
    - Prevención de sesgos temporales (barajado estocástico interno).
    """
    def __init__(self, df_map, train_set_paths, estimator, scoring_func,
                 feature_cols, target_col, label_mapping=None, secondary_scoring_funcs=None):
        self.df_map = df_map if isinstance(df_map, pd.DataFrame) else self._parse_map(df_map)
        self.train_set_paths = train_set_paths
        self.estimator = estimator
        self.scoring_func = scoring_func
        self.secondary_scoring_funcs = secondary_scoring_funcs or {}
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.label_mapping = label_mapping
        self.patient_cache = {}
        self.prep_cache = {}
        self.tqdm_format = "{l_bar}{bar:20}{r_bar}"
        self.last_score = 0.0

    def _parse_map(self, d):
        rows = []
        for f_id, data in d.items():
            for p_id in data.get('patients', []):
                rows.append({'fold_id': f_id, 'patient_num_id': str(p_id)})
        return pd.DataFrame(rows)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _validate_inputs(self, scaler_type):
        valid_scalers = ['robust', 'standard', 'quantile_uniform', 'quantile_gaussian', 'none']
        if scaler_type not in valid_scalers:
            raise ValueError(f"scaler_type '{scaler_type}' no soportado.")

    def _check_pca_warning(self, spaces, scaler_type):
        if scaler_type != 'none' or not spaces:
            return
        has_pca = False
        for space in spaces.values():
            space_list = [space] if isinstance(space, dict) else space
            for s in space_list:
                if 'pca__n_components' in s:
                    pca_vals = s['pca__n_components']
                    if isinstance(pca_vals, list) and any(v is not None for v in pca_vals):
                        has_pca = True
                        break
            if has_pca: break
        if has_pca:
            warnings.warn("PCA detectado sin escalado ('none'). Riesgo de sesgo por magnitudes.", UserWarning)

    def pre_cache_data(self):
        target_ids = set(self.df_map['patient_num_id'].unique())
        paths = [p for p in self.train_set_paths if p.name in target_ids]
        cols = self.feature_cols + [self.target_col]
        pbar = tqdm(paths, desc="Data Caching")
        for p_path in pbar:
            x_pts, y_pts = [] , []
            for f in p_path.glob("*.parquet"):
                df = pd.read_parquet(f, columns=cols)
                if self.label_mapping:
                    df[self.target_col] = df[self.target_col].astype(str).str.lower().str.strip().map(self.label_mapping)
                df = df.dropna(subset=[self.target_col])
                if len(df) == 0: continue
                x_pts.append(df[self.feature_cols].values.astype(np.float32))
                y_pts.append(df[self.target_col].values.astype(np.int8))
            if x_pts:
                self.patient_cache[p_path.name] = (np.vstack(x_pts), np.concatenate(y_pts))
        gc.collect()

    def fit(self, methods=['grid'], spaces=None, n_trials=10, patience=15, tol=1e-3, samples_per_fold=40000,
            scaler_type='robust', seed=42, checkpoint_dir=None, checkpoint_level='pca'):

        if spaces and not any(isinstance(v, (dict, list)) for v in spaces.values()):
            spaces = {m: spaces for m in methods}

        is_gnb = type(self.estimator).__name__ == 'GaussianNBModel'
        is_knn = type(self.estimator).__name__ == 'KNNModel'

        if is_gnb:
            if scaler_type != 'quantile_gaussian':
                print("\n[INFO] GaussianNBModel detectado. Forzando quantile_gaussian.")
                scaler_type = 'quantile_gaussian'
            if spaces:
                for m, space in spaces.items():
                    space_list = [space] if isinstance(space, dict) else space
                    for s in space_list:
                        bal_options = s.get('balancing', ['none'])
                        bal_options = [bal_options] if not isinstance(bal_options, list) else bal_options
                        if any(b in ['SMOTE', 'ADASYN'] for b in bal_options):
                            raise ValueError("GaussianNBModel no permite SMOTE/ADASYN.")
        if is_knn and spaces:
            for m, space in spaces.items():
                space_list = [space] if isinstance(space, dict) else space
                for s in space_list:
                    if 'weight_balanced' in s.get('balancing', ['none']):
                        raise ValueError("KNNModel no acepta sample_weight.")

        self._validate_inputs(scaler_type)
        self._check_pca_warning(spaces, scaler_type)

        if not self.patient_cache: self.pre_cache_data()
        self._set_seed(seed)

        dict_results = {}
        dict_convergence = {}
        benchmark_rows = []
        folds = sorted(self.df_map['fold_id'].unique())

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        for method in methods:
            if method not in spaces: continue
            current_space = spaces[method]
            self.last_score = 0.0
            self.prep_cache.clear()
            gc.collect()

            chkpt_path = os.path.join(checkpoint_dir, f"checkpoint_{method}") if checkpoint_dir else None

            if method == 'grid':
                res, evals, conv, early_stop, exec_time = self._grid_search(current_space, folds, samples_per_fold, scaler_type, seed, chkpt_path, checkpoint_level)
            elif method == 'random':
                res, evals, conv, early_stop, exec_time = self._random_search(current_space, n_trials, folds, samples_per_fold, scaler_type, seed, chkpt_path)
            elif method == 'optuna':
                res, evals, conv, early_stop, exec_time = self._optuna_search(current_space, n_trials, folds, samples_per_fold, scaler_type, seed, patience=patience, tol=tol, checkpoint_path=chkpt_path)
            elif method == 'genetic':
                res, evals, conv, early_stop, exec_time = self._genetic_search(current_space, n_trials, folds, samples_per_fold, scaler_type, seed, patience=patience, tol=tol, checkpoint_path=chkpt_path)
            else:
                continue

            df_res = pd.DataFrame(res)
            if not df_res.empty:
                best_idx = df_res['primary_center'].idxmax()
                best_center = df_res.loc[best_idx, 'primary_center']
                best_spread = df_res.loc[best_idx, 'primary_spread']
            else:
                best_center, best_spread = 0.0, 0.0

            dict_results[method] = df_res
            dict_convergence[method] = conv
            benchmark_rows.append({
                'optimization_method': method,
                'execution_time_sec': exec_time,
                'total_evaluations': evals,
                'best_score_center': best_center,
                'best_score_spread': best_spread,
                'early_stopped': early_stop
            })
            self.prep_cache.clear()
            gc.collect()

        return dict_results, pd.DataFrame(benchmark_rows), dict_convergence

    # -------------------------------------------------------------------------
    # Grid Search
    # -------------------------------------------------------------------------
    def _grid_search(self, space, folds, samples, scaler_type, seed, checkpoint_path=None, checkpoint_level='pca'):
        start_time = time.perf_counter()
        raw_grid = list(ParameterGrid(space))
        grid = []
        for cfg in raw_grid:
            cfg = dict(cfg)
            v = cfg.get('pca__n_components', 0)
            cfg['pca__n_components'] = 0 if (v is None or str(v).lower() == 'none') else int(v)
            grid.append(cfg)

        pca_vals = {cfg['pca__n_components'] for cfg in grid}
        unique_ks = sorted(list(pca_vals))
        bal_vals = {cfg.get('balancing', 'none') for cfg in grid}
        unique_bal = list(bal_vals)

        is_gnb = type(self.estimator).__name__ == 'GaussianNBModel'
        is_knn = type(self.estimator).__name__ == 'KNNModel'
        inf_params = ['decision_strategy', 'voting_weight'] if (is_gnb or is_knn) else []
        if is_knn: inf_params.append('n_neighbors')

        sec_keys = list(self.secondary_scoring_funcs.keys())

        fold_accumulator = {
            json.dumps(cfg, sort_keys=True): {
                'primary': {},
                'secondary': {m: {} for m in sec_keys}
            } for cfg in grid
        }

        conv_vector, eval_count = [], 0
        cumulative_time = 0.0
        completed_folds, completed_pca, completed_bal = [], [], []

        def parse_k(k): return 0 if (k is None or str(k).lower() == 'none') else int(k)

        def save_grid_chkpt():
            current_elapsed = cumulative_time + (time.perf_counter() - start_time)
            if checkpoint_path:
                tmp_path = f"{checkpoint_path}.tmp"
                with open(tmp_path, 'w') as f:
                    json.dump({
                        'space': space,
                        'completed_folds': completed_folds,
                        'completed_pca': completed_pca,
                        'completed_bal': completed_bal,
                        'fold_accumulator': fold_accumulator,
                        'eval_count': eval_count,
                        'cumulative_time': current_elapsed
                    }, f, indent=4, cls=NumpyEncoder)
                os.replace(tmp_path, f"{checkpoint_path}.json")

        if checkpoint_path and os.path.exists(f"{checkpoint_path}.json"):
            try:
                with open(f"{checkpoint_path}.json", 'r') as f:
                    chkpt = json.load(f)
                if json.dumps(chkpt.get('space', {}), sort_keys=True, cls=NumpyEncoder) == json.dumps(space, sort_keys=True, cls=NumpyEncoder):
                    completed_folds = [int(f) for f in chkpt.get('completed_folds', [])]
                    completed_pca = [[int(f), parse_k(k)] for f, k in chkpt.get('completed_pca', [])]
                    completed_bal = [[int(f), parse_k(k), str(b)] for f, k, b in chkpt.get('completed_bal', [])]
                    fold_accumulator = chkpt['fold_accumulator']
                    eval_count = chkpt.get('eval_count', 0)
                    cumulative_time = chkpt.get('cumulative_time', 0.0)
            except Exception as e:
                print(f"[ERROR] Checkpoint grid: {e}")

        remaining_folds = [f for f in folds if int(f) not in completed_folds]
        pbar_folds = tqdm(remaining_folds, desc="Folds Global", position=0, leave=True, total=len(folds))

        for f_id in pbar_folds:
            fold_hash = int(hashlib.md5(str(f_id).encode()).hexdigest(), 16) % (10**8) if f_id is not None else 0
            effective_seed = seed ^ fold_hash
            X_tr_base, y_tr_base = self._stratified_subsample(
                self.df_map[self.df_map['fold_id'] != f_id]['patient_num_id'], samples, effective_seed
            )
            X_vl_base, y_vl_base = self._get_val_data(self.df_map[self.df_map['fold_id'] == f_id]['patient_num_id'])

            if len(X_tr_base) == 0:
                warnings.warn(f"Fold {f_id}: no hay datos de entrenamiento. Asignando NaN a todas las configuraciones.")
                for cfg in grid:
                    cfg_key = json.dumps(cfg, sort_keys=True)
                    fold_accumulator[cfg_key]['primary'][str(f_id)] = np.nan
                    for n in sec_keys:
                        fold_accumulator[cfg_key]['secondary'][n][str(f_id)] = np.nan
                completed_folds.append(int(f_id))
                save_grid_chkpt()
                continue

            remaining_ks = [k for k in unique_ks if [int(f_id), parse_k(k)] not in completed_pca]
            pbar_pca = tqdm(remaining_ks, desc=" -> PCA Dim", position=1, leave=False, total=len(unique_ks))

            for k in pbar_pca:
                remaining_bals = [b for b in unique_bal if [int(f_id), parse_k(k), str(b)] not in completed_bal]
                if not remaining_bals: continue

                sc = self._get_scaler(scaler_type, len(X_tr_base), effective_seed)
                X_tr_step = sc.fit_transform(X_tr_base)
                X_vl_step = sc.transform(X_vl_base)

                if k:
                    pca = PCA(n_components=k, svd_solver='randomized', random_state=effective_seed)
                    X_tr_step = pca.fit_transform(X_tr_step)
                    X_vl_step = pca.transform(X_vl_step)

                for bal in remaining_bals:
                    X_tr_bal, y_tr_bal, sw = X_tr_step, y_tr_base, None
                    
                    if bal in ['SMOTE', 'ADASYN']:
                        min_samples = np.min(np.unique(y_tr_base, return_counts=True)[1])
                        k_n = min(5, min_samples - 1) if min_samples > 1 else 1
                        if bal == 'SMOTE': 
                            X_tr_bal, y_tr_bal = SMOTE(random_state=effective_seed, k_neighbors=k_n).fit_resample(X_tr_step, y_tr_base)
                        elif bal == 'ADASYN': 
                            X_tr_bal, y_tr_bal = ADASYN(random_state=effective_seed, n_neighbors=k_n).fit_resample(X_tr_step, y_tr_base)
                    elif bal == 'weight_balanced': 
                        sw = compute_sample_weight('balanced', y_tr_base)

                    sub_grid = [c for c in grid if c.get('pca__n_components') == k and c.get('balancing', 'none') == bal]
                    with tqdm(total=len(sub_grid), desc="      -> Hyperparams", position=2, leave=False) as pbar_cfg:
                        for cfg in sub_grid:
                            eval_count += 1
                            fit_kwargs = {}
                            predict_kwargs = {}
                            for key, val in cfg.items():
                                if 'pca__' in key: continue
                                if key == 'balancing': continue
                                param_name = key.split('__')[-1]
                                if param_name in inf_params:
                                    predict_kwargs[param_name] = val
                                else:
                                    fit_kwargs[param_name] = val

                            model = clone(self.estimator).set_params(**fit_kwargs)
                            success = False
                            for attempt in range(3):
                                if cp is not None:
                                    cp.get_default_memory_pool().free_all_blocks()
                                try:
                                    if is_knn: model.fit(X_tr_bal, y_tr_bal)
                                    else: model.fit(X_tr_bal, y_tr_bal, sample_weight=sw)
                                    success = True
                                    break
                                except Exception:
                                    if attempt < 2: time.sleep(1.5)

                            cfg_key = json.dumps(cfg, sort_keys=True)
                            f_key = str(f_id)

                            if not success:
                                fold_accumulator[cfg_key]['primary'][f_key] = np.nan
                                for n in sec_keys:
                                    fold_accumulator[cfg_key]['secondary'][n][f_key] = np.nan
                                pbar_cfg.update(1)
                                continue

                            try:
                                y_pred = model.predict(X_vl_step, **predict_kwargs)
                                score = self.scoring_func(y_vl_base, y_pred)
                                sec_scores = {n: f(y_vl_base, y_pred) for n, f in self.secondary_scoring_funcs.items()}
                            except Exception as e:
                                print(f"\n[ERROR] Predict falló: {e}. Asignando NaN.")
                                score = np.nan
                                sec_scores = {n: np.nan for n in sec_keys}

                            fold_accumulator[cfg_key]['primary'][f_key] = score
                            for n in sec_keys:
                                fold_accumulator[cfg_key]['secondary'][n][f_key] = sec_scores.get(n, np.nan)
                            self.last_score = score if not np.isnan(score) else self.last_score
                            pbar_cfg.set_postfix({"score": f"{score:.4f}" if not np.isnan(score) else "NaN"})
                            pbar_cfg.update(1)

                    if checkpoint_level == 'balancing':
                        completed_bal.append([int(f_id), parse_k(k), str(bal)])
                        save_grid_chkpt()
                if checkpoint_level in ['pca', 'balancing']:
                    completed_pca.append([int(f_id), parse_k(k)])
                    save_grid_chkpt()
            completed_folds.append(int(f_id))
            save_grid_chkpt()

        results = []
        for cfg in grid:
            cfg_key = json.dumps(cfg, sort_keys=True)
            prim_dict = fold_accumulator[cfg_key]['primary']
            sec_dict = fold_accumulator[cfg_key]['secondary']
            primary_vals = np.array([prim_dict[f] for f in prim_dict], dtype=float)
            if len(primary_vals) == 0 or np.all(np.isnan(primary_vals)):
                p_center, p_spread = np.nan, np.nan
            else:
                p_center, p_spread = np.nanmedian(primary_vals), np.nanstd(primary_vals)

            res = {'params': cfg_key, 'balancing': cfg.get('balancing', 'none'),
                   'primary_center': p_center, 'primary_spread': p_spread}
            for m in sec_keys:
                sec_folds = sec_dict[m]
                sec_vals = np.array([sec_folds[f] for f in sec_folds], dtype=float)
                if np.all(np.isnan(sec_vals)):
                    if 'pval' in m: res[f'{m}_pass_rate'] = np.nan
                    res[f'{m}_center'], res[f'{m}_spread'] = np.nan, np.nan
                else:
                    if 'pval' in m: res[f'{m}_pass_rate'] = np.nanmean(sec_vals > 0.05)
                    res[f'{m}_center'] = np.nanmedian(sec_vals)
                    res[f'{m}_spread'] = np.nanstd(sec_vals)
            results.append(res)
            conv_vector.append({'center': p_center, 'spread': p_spread})

        return results, eval_count, conv_vector, np.nan, cumulative_time + (time.perf_counter() - start_time)

    # -------------------------------------------------------------------------
    # Optuna Search
    # -------------------------------------------------------------------------
    def _optuna_search(self, space, trials, folds, samples, scaler_type, seed, patience=15, tol=1e-3, checkpoint_path=None):
        start_time = time.perf_counter()
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        results, conv_vector, evals = [], [], [0]
        stagnant_trials, best_historical_score, early_stopped, cumulative_time = [0], [-float('inf')], [False], [0.0]
        replay_cache = {}

        if checkpoint_path and os.path.exists(f"{checkpoint_path}.json"):
            try:
                with open(f"{checkpoint_path}.json", 'r') as f: chkpt = json.load(f)
                if json.dumps(chkpt.get('space', {}), sort_keys=True, cls=NumpyEncoder) == json.dumps(space, sort_keys=True, cls=NumpyEncoder):
                    for pe in chkpt.get('past_evaluations', []):
                        study.enqueue_trial(pe['cfg'])
                        replay_cache[json.dumps(pe['cfg'], sort_keys=True, cls=NumpyEncoder)] = pe
                        results.append(pe['res'])
                        conv_vector.append({'center': pe['res']['primary_center'], 'spread': pe['res']['primary_spread']})
                    evals[0] = chkpt.get('evals', 0)
                    best_historical_score[0] = chkpt.get('best_historical_score', -float('inf'))
                    stagnant_trials[0] = chkpt.get('stagnant_trials', 0)
                    cumulative_time[0] = chkpt.get('cumulative_time', 0.0)
            except Exception as e: print(f"[ERROR] Checkpoint Optuna: {e}")

        pbar = tqdm(total=trials, desc="OptunaSearch", initial=len(replay_cache))

        def objective(trial):
            cfg = {}
            for k, v in space.items():
                if isinstance(v, list): cfg[k] = trial.suggest_categorical(k, v)
                elif isinstance(v, tuple):
                    if len(v) == 3 and v[2] == 'log': cfg[k] = trial.suggest_float(k, v[0], v[1], log=True)
                    elif len(v) == 3 and v[2] == 'int': cfg[k] = trial.suggest_int(k, v[0], v[1])
                    else: cfg[k] = trial.suggest_float(k, v[0], v[1])
                else: cfg[k] = v
            cfg_str = json.dumps(cfg, sort_keys=True, cls=NumpyEncoder)
            if cfg_str in replay_cache:
                self.last_score = replay_cache[cfg_str]['res']['primary_center']
                pbar.set_postfix({"score": f"{self.last_score:.4f}"})
                pbar.update(1)
                return self.last_score

            f_scores, f_sec = [], []
            for f_id in folds:
                evals[0] += 1
                s, sec = self._atomic_evaluation(cfg, f_id, samples, scaler_type, seed)
                if np.isnan(s): return -9999.0
                f_scores.append(s); f_sec.append(sec)

            res = self._consolidate(cfg, f_scores, f_sec)
            self.last_score = res['primary_center']
            results.append(res)
            conv_vector.append({'center': res['primary_center'], 'spread': res['primary_spread']})
            pbar.set_postfix({"score": f"{self.last_score:.4f}"})
            pbar.update(1)

            if checkpoint_path:
                current_elapsed = cumulative_time[0] + (time.perf_counter() - start_time)
                replay_cache[cfg_str] = {'cfg': cfg, 'res': res, 'evals': evals[0]}
                with open(f"{checkpoint_path}.tmp", 'w') as f:
                    json.dump({
                        'space': space, 'past_evaluations': list(replay_cache.values()),
                        'evals': evals[0], 'best_historical_score': best_historical_score[0],
                        'stagnant_trials': stagnant_trials[0], 'cumulative_time': current_elapsed
                    }, f, indent=4, cls=NumpyEncoder)
                os.replace(f"{checkpoint_path}.tmp", f"{checkpoint_path}.json")
            return self.last_score

        def early_stopping_callback(study, trial):
            if study.best_value - best_historical_score[0] > tol:
                best_historical_score[0] = study.best_value
                stagnant_trials[0] = 0
            else:
                stagnant_trials[0] += 1
            if stagnant_trials[0] >= patience:
                early_stopped[0] = True
                study.stop()

        if trials > 0: study.optimize(objective, n_trials=trials, callbacks=[early_stopping_callback])
        pbar.close()
        return results, evals[0], conv_vector, early_stopped[0], cumulative_time[0] + (time.perf_counter() - start_time)

    # -------------------------------------------------------------------------
    # Random Search
    # -------------------------------------------------------------------------
    def _random_search(self, space, trials, folds, samples, scaler_type, seed, checkpoint_path=None):
        start_time = time.perf_counter()
        parsed_space = {}
        for k, v in space.items():
            if isinstance(v, list): parsed_space[k] = v
            elif isinstance(v, tuple):
                if len(v) == 3 and v[2] == 'log': parsed_space[k] = stats.loguniform(v[0], v[1])
                elif len(v) == 3 and v[2] == 'int': parsed_space[k] = stats.randint(v[0], v[1] + 1)
                else: parsed_space[k] = stats.uniform(loc=v[0], scale=v[1]-v[0])
            else: parsed_space[k] = [v]

        sampler = list(ParameterSampler(parsed_space, n_iter=trials, random_state=seed))
        results, conv_vector, evals = [], [], 0
        cumulative_time = 0.0
        completed_configs = []

        if checkpoint_path and os.path.exists(f"{checkpoint_path}.json"):
            try:
                with open(f"{checkpoint_path}.json", 'r') as f: chkpt = json.load(f)
                if json.dumps(chkpt.get('space', {}), sort_keys=True, cls=NumpyEncoder) == json.dumps(space, sort_keys=True, cls=NumpyEncoder):
                    results = chkpt['results']
                    conv_vector = chkpt['conv_vector']
                    evals = chkpt.get('evals', 0)
                    cumulative_time = chkpt.get('cumulative_time', 0.0)
                    completed_configs = chkpt['completed_configs']
            except Exception as e: print(f"[ERROR] Checkpoint Random: {e}")

        completed_configs_set = set(completed_configs)
        remaining_configs = [cfg for cfg in sampler if json.dumps(cfg, sort_keys=True, cls=NumpyEncoder) not in completed_configs_set]
        pbar = tqdm(remaining_configs, desc="RandomSearch", initial=len(completed_configs), total=trials)

        for cfg in pbar:
            f_scores, f_sec = [], []
            skip = False
            cfg_str = json.dumps(cfg, sort_keys=True, cls=NumpyEncoder)
            for f_id in folds:
                evals += 1
                s, sec = self._atomic_evaluation(cfg, f_id, samples, scaler_type, seed)
                if np.isnan(s):
                    skip = True
                    break
                f_scores.append(s); f_sec.append(sec)
            if skip: continue
            res = self._consolidate(cfg, f_scores, f_sec)
            self.last_score = res['primary_center']
            pbar.set_postfix({"score": f"{self.last_score:.4f}"})
            results.append(res)
            conv_vector.append({'center': res['primary_center'], 'spread': res['primary_spread']})
            completed_configs.append(cfg_str)

            if checkpoint_path:
                current_elapsed = cumulative_time + (time.perf_counter() - start_time)
                with open(f"{checkpoint_path}.tmp", 'w') as f:
                    json.dump({
                        'space': space, 'results': results, 'conv_vector': conv_vector,
                        'evals': evals, 'completed_configs': completed_configs, 'cumulative_time': current_elapsed
                    }, f, indent=4, cls=NumpyEncoder)
                os.replace(f"{checkpoint_path}.tmp", f"{checkpoint_path}.json")

        return results, evals, conv_vector, np.nan, cumulative_time + (time.perf_counter() - start_time)

    # -------------------------------------------------------------------------
    # Genetic Search
    # -------------------------------------------------------------------------
    def _genetic_search(self, space, n_trials, folds, samples, scaler_type, seed, pop_size=5, patience=5, tol=1e-3, checkpoint_path=None):
        start_time = time.perf_counter()
        self._set_seed(seed)
        keys = list(space.keys())

        def sample_param(v):
            if isinstance(v, list): return random.choice(v)
            if isinstance(v, tuple):
                if len(v) == 3 and v[2] == 'log': return np.exp(random.uniform(np.log(v[0]), np.log(v[1])))
                if len(v) == 3 and v[2] == 'int': return random.randint(v[0], v[1])
                return random.uniform(v[0], v[1])
            return v

        pop = [{k: sample_param(space[k]) for k in keys} for _ in range(pop_size)]
        results, conv_vector, evals = [], [], 0
        cumulative_time = 0.0
        generations = max(1, n_trials // pop_size)
        prev_gen_best = None
        stagnant_generations = 0
        early_stopped = False
        start_gen = 0

        if checkpoint_path and os.path.exists(f"{checkpoint_path}.json"):
            try:
                with open(f"{checkpoint_path}.json", 'r') as f: chkpt = json.load(f)
                if json.dumps(chkpt.get('space', {}), sort_keys=True, cls=NumpyEncoder) == json.dumps(space, sort_keys=True, cls=NumpyEncoder):
                    pop = chkpt['pop']
                    results = chkpt['results']
                    conv_vector = chkpt['conv_vector']
                    evals = chkpt.get('evals', 0)
                    cumulative_time = chkpt.get('cumulative_time', 0.0)
                    prev_gen_best = chkpt['prev_gen_best']
                    stagnant_generations = chkpt['stagnant_generations']
                    start_gen = chkpt['start_gen']
            except Exception as e: print(f"[ERROR] Checkpoint Genetic: {e}")

        pbar = tqdm(range(start_gen, generations), desc="GeneticSearch", initial=start_gen, total=generations)
        
        completed_cfgs = {r['params']: r for r in results}

        for gen in pbar:
            gen_scores = []
            valid_pop = []
            for cfg in pop:
                cfg_str = json.dumps(cfg, sort_keys=True, cls=NumpyEncoder)
                
                if cfg_str in completed_cfgs:
                    res = completed_cfgs[cfg_str]
                    self.last_score = res['primary_center']
                    gen_scores.append(self.last_score)
                    valid_pop.append(cfg)
                    continue 

                f_scores, f_sec = [], []
                skip = False
                for f_id in folds:
                    evals += 1
                    s, sec = self._atomic_evaluation(cfg, f_id, samples, scaler_type, seed)
                    if np.isnan(s):
                        skip = True
                        break
                    f_scores.append(s); f_sec.append(sec)
                if skip: continue
                res = self._consolidate(cfg, f_scores, f_sec)
                self.last_score = res['primary_center']
                results.append(res)
                completed_cfgs[cfg_str] = res
                conv_vector.append({'center': res['primary_center'], 'spread': res['primary_spread']})
                gen_scores.append(self.last_score)
                valid_pop.append(cfg)

            if len(valid_pop) < 2:
                pbar.write("[ALERTA] Menos de 2 individuos válidos. Reiniciando población.")
                pop = [{k: sample_param(space[k]) for k in keys} for _ in range(pop_size)]
                continue

            gen_best = max(gen_scores)
            pbar.set_postfix({"best_gen": f"{gen_best:.4f}"})

            if prev_gen_best is None:
                prev_gen_best = gen_best
                stagnant_generations = 0
            else:
                if (gen_best - prev_gen_best) > tol:
                    prev_gen_best = gen_best
                    stagnant_generations = 0
                else:
                    stagnant_generations += 1
            if stagnant_generations >= patience:
                early_stopped = True
                break

            best_idx = np.argsort(gen_scores)[-2:]
            parents = [valid_pop[i] for i in best_idx] 

            new_pop = [copy.deepcopy(parents[-1])]  
            for _ in range(pop_size - 1):
                child = {}
                for k in keys:
                    if random.random() < 0.05: 
                        child[k] = sample_param(space[k])
                    else:
                        child[k] = random.choice(parents)[k] 
                new_pop.append(child)
            pop = new_pop

            if checkpoint_path:
                current_elapsed = cumulative_time + (time.perf_counter() - start_time)
                with open(f"{checkpoint_path}.tmp", 'w') as f:
                    json.dump({
                        'space': space, 'pop': pop, 'results': results, 'conv_vector': conv_vector,
                        'evals': evals, 'prev_gen_best': prev_gen_best, 'stagnant_generations': stagnant_generations,
                        'start_gen': gen + 1, 'cumulative_time': current_elapsed
                    }, f, indent=4, cls=NumpyEncoder)
                os.replace(f"{checkpoint_path}.tmp", f"{checkpoint_path}.json")

        return results, evals, conv_vector, early_stopped, cumulative_time + (time.perf_counter() - start_time)

    # -------------------------------------------------------------------------
    # Evaluación atómica
    # -------------------------------------------------------------------------
    def _atomic_evaluation(self, cfg, f_id, samples, stype, seed):
        k = cfg.get('pca__n_components', 0)
        k = 0 if (k is None or (isinstance(k, str) and k.lower() == 'none')) else int(k)
        bal = cfg.get('balancing', 'none')
        is_gnb = type(self.estimator).__name__ == 'GaussianNBModel'
        is_knn = type(self.estimator).__name__ == 'KNNModel'
        inf_params = ['decision_strategy', 'voting_weight'] if (is_gnb or is_knn) else []
        if is_knn: inf_params.append('n_neighbors')

        fold_hash = int(hashlib.md5(str(f_id).encode()).hexdigest(), 16) % (10**8) if f_id is not None else 0
        effective_seed = seed ^ fold_hash

        cache_key = (f_id, k, bal, stype, samples, effective_seed)
        if cache_key not in self.prep_cache:
            if len(self.prep_cache) >= 50:
                oldest_key = next(iter(self.prep_cache))
                del self.prep_cache[oldest_key]
            
            X_tr, y_tr = self._stratified_subsample(
                self.df_map[self.df_map['fold_id'] != f_id]['patient_num_id'], samples, effective_seed
            )
            X_vl, y_vl = self._get_val_data(self.df_map[self.df_map['fold_id'] == f_id]['patient_num_id'])

            if len(X_tr) == 0:
                self.prep_cache[cache_key] = (None, None, None, None, None)
                return np.nan, {n: np.nan for n in self.secondary_scoring_funcs.keys()}

            sc = self._get_scaler(stype, len(X_tr), effective_seed)
            X_tr, X_vl = sc.fit_transform(X_tr), sc.transform(X_vl)
            if k:
                pca = PCA(n_components=k, svd_solver='randomized', random_state=effective_seed)
                X_tr, X_vl = pca.fit_transform(X_tr), pca.transform(X_vl)

            sw = compute_sample_weight('balanced', y_tr) if bal == 'weight_balanced' else None
            
            if bal in ['SMOTE', 'ADASYN']:
                min_samples = np.min(np.unique(y_tr, return_counts=True)[1])
                k_n = min(5, min_samples - 1) if min_samples > 1 else 1
                if bal == 'SMOTE': 
                    X_tr, y_tr = SMOTE(random_state=effective_seed, k_neighbors=k_n).fit_resample(X_tr, y_tr)
                elif bal == 'ADASYN': 
                    X_tr, y_tr = ADASYN(random_state=effective_seed, n_neighbors=k_n).fit_resample(X_tr, y_tr)

            self.prep_cache[cache_key] = (X_tr, y_tr, X_vl, y_vl, sw)

        X_t, y_t, X_v, y_v, sw_c = self.prep_cache[cache_key]
        if X_t is None:
            return np.nan, {n: np.nan for n in self.secondary_scoring_funcs.keys()}

        fit_kwargs = {}
        predict_kwargs = {}
        for key, val in cfg.items():
            if 'pca__' in key: continue
            if key == 'balancing': continue
            param_name = key.split('__')[-1]
            if param_name in inf_params:
                predict_kwargs[param_name] = val
            else:
                fit_kwargs[param_name] = val

        model = clone(self.estimator).set_params(**fit_kwargs)

        success = False
        for attempt in range(3):
            if cp is not None:
                cp.get_default_memory_pool().free_all_blocks()
            try:
                if is_knn: model.fit(X_t, y_t)
                else: model.fit(X_t, y_t, sample_weight=sw_c)
                success = True
                break
            except Exception:
                if attempt < 2: time.sleep(1.5)

        if not success:
            return np.nan, {n: np.nan for n in self.secondary_scoring_funcs.keys()}

        try:
            y_pred = model.predict(X_v, **predict_kwargs)
            score = self.scoring_func(y_v, y_pred)
            sec_res = {n: f(y_v, y_pred) for n, f in self.secondary_scoring_funcs.items()}
        except Exception as e:
            print(f"\n[ERROR] Predict falló en evaluación atómica: {e}. Asignando NaN.")
            score = np.nan
            sec_res = {n: np.nan for n in self.secondary_scoring_funcs.keys()}

        return score, sec_res

    # -------------------------------------------------------------------------
    # Submuestreo estratificado (con validación de proporción < 2%)
    # -------------------------------------------------------------------------
    def _stratified_subsample(self, ids, n, seed):
        """
        Submuestreo que intenta representar a todos los pacientes.
        Valida que la proporción de clases no difiera en más del 2% respecto a la original.
        Recibe una semilla que ya debe venir modificada por el fold.
        Aplica barajado final obligatorio para algoritmos sensibles al orden.
        """
        ids_s = [str(i) for i in ids if str(i) in self.patient_cache]
        if not ids_s:
            return np.empty((0, len(self.feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int8)

        rng = np.random.RandomState(seed)

        # Datos completos del subset
        X_full, y_full = [], []
        for p in ids_s:
            X, y = self.patient_cache[p]
            X_full.append(X)
            y_full.append(y)
        X_full = np.vstack(X_full)
        y_full = np.concatenate(y_full)

        # Proporciones originales
        classes_orig, counts_orig = np.unique(y_full, return_counts=True)
        total_orig = len(y_full)
        orig_props = {cls: cnt / total_orig for cls, cnt in zip(classes_orig, counts_orig)}

        if n is None or n >= total_orig:
            # BLINDAJE 1: Barajar incluso si se usan todos los datos
            shuffler = rng.permutation(total_orig)
            return X_full[shuffler], y_full[shuffler]

        # Cuotas exactas
        target_counts = {}
        for cls in classes_orig:
            target_counts[cls] = max(1, int(round(orig_props[cls] * n)))

        # Ajuste para que sumen exactamente n
        diff = n - sum(target_counts.values())
        fracs = {cls: (orig_props[cls] * n) - target_counts[cls] for cls in classes_orig}
        sorted_cls = sorted(classes_orig, key=lambda c: fracs[c], reverse=True)
        for i in range(abs(diff)):
            idx = i % len(sorted_cls)
            if diff > 0:
                target_counts[sorted_cls[idx]] += 1
            else:
                target_counts[sorted_cls[-1 - idx]] -= 1

        final_indices = []
        total_sampled = 0
        for cls in classes_orig:
            target = target_counts[cls]
            patients_with_cls = []
            counts_per_patient = []
            for p in ids_s:
                y_p = self.patient_cache[p][1]
                cnt = np.sum(y_p == cls)
                if cnt > 0:
                    patients_with_cls.append(p)
                    counts_per_patient.append(cnt)

            if not patients_with_cls:
                continue

            total_cls_available = sum(counts_per_patient)
            min_alloc = [1] * len(patients_with_cls)

            if target < len(min_alloc):
                warnings.warn(
                    f"Clase {cls}: objetivo ({target}) < pacientes ({len(patients_with_cls)}). "
                    "Se seleccionarán pacientes al azar.", RuntimeWarning
                )
                chosen = rng.choice(len(patients_with_cls), size=target, replace=False)
                alloc = [1 if i in chosen else 0 for i in range(len(patients_with_cls))]
            else:
                alloc = min_alloc.copy()
                remaining = target - sum(alloc)
                if remaining > 0 and total_cls_available > 0:
                    weights = [cnt / total_cls_available for cnt in counts_per_patient]
                    raw = [w * remaining for w in weights]
                    floor_raw = [int(r) for r in raw]
                    frac_remain = remaining - sum(floor_raw)
                    frac_pairs = sorted([(raw[i] - floor_raw[i], i) for i in range(len(patients_with_cls))], reverse=True)
                    for _, i in frac_pairs[:frac_remain]:
                        floor_raw[i] += 1
                    alloc = [min_alloc[i] + floor_raw[i] for i in range(len(patients_with_cls))]

                for i, cnt in enumerate(counts_per_patient):
                    alloc[i] = min(alloc[i], cnt)
                actual_total = sum(alloc)
                if actual_total < target:
                    deficit = target - actual_total
                    available = [(cnt - alloc[i], i) for i, cnt in enumerate(counts_per_patient) if cnt > alloc[i]]
                    available.sort(reverse=True)
                    for avail, i in available:
                        take = min(deficit, avail)
                        alloc[i] += take
                        deficit -= take
                        if deficit == 0:
                            break
                    if deficit > 0:
                        warnings.warn(
                            f"Clase {cls}: déficit de {deficit} muestras. "
                            "Se usará el máximo disponible.", RuntimeWarning
                        )

            for p, a in zip(patients_with_cls, alloc):
                if a == 0:
                    continue
                X_p, y_p = self.patient_cache[p]
                idx_cls = np.where(y_p == cls)[0]
                chosen = rng.choice(idx_cls, size=a, replace=False)
                final_indices.append((p, chosen))
                total_sampled += a

        if total_sampled < n:
            warnings.warn(
                f"Submuestreo total: {total_sampled} muestras (objetivo: {n}). "
                "Faltan datos para alcanzar el tamaño solicitado.", RuntimeWarning
            )

        X_out, y_out = [], []
        for p, idx in final_indices:
            X_p, y_p = self.patient_cache[p]
            X_out.append(X_p[idx])
            y_out.append(y_p[idx])
            
        if X_out:
            X_sub = np.vstack(X_out)
            y_sub = np.concatenate(y_out)
            
            # BLINDAJE 2: Destrucción del orden contiguo por pacientes antes del return
            shuffler = rng.permutation(len(y_sub))
            X_sub = X_sub[shuffler]
            y_sub = y_sub[shuffler]
        else:
            X_sub = np.empty((0, len(self.feature_cols)), dtype=np.float32)
            y_sub = np.empty((0,), dtype=np.int8)

        # Validación explícita
        if len(y_sub) > 0:
            classes_sub, counts_sub = np.unique(y_sub, return_counts=True)
            sub_props = {cls: cnt / len(y_sub) for cls, cnt in zip(classes_sub, counts_sub)}
            for cls in classes_orig:
                original_prop = orig_props[cls]
                subsampled_prop = sub_props.get(cls, 0.0)
                deviation = abs(original_prop - subsampled_prop)
                if deviation > 0.02:
                    warnings.warn(
                        f"¡Desviación > 2% en clase {cls}! Original: {original_prop:.4f}, "
                        f"Submuestreo: {subsampled_prop:.4f} (Δ={deviation:.4f}).",
                        RuntimeWarning
                    )

        return X_sub, y_sub

    # -------------------------------------------------------------------------
    # Utilidades
    # -------------------------------------------------------------------------
    def _get_val_data(self, ids):
        x, y = [], []
        for i in ids:
            if str(i) in self.patient_cache:
                d = self.patient_cache[str(i)]
                x.append(d[0]); y.append(d[1])
        if not x:
            return np.empty((0, len(self.feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int8)
        return np.vstack(x), np.concatenate(y)

    def _get_scaler(self, scaler_type, n_samples, seed):
        if scaler_type == 'none': return FunctionTransformer(func=None)
        elif scaler_type == 'quantile_uniform': return QuantileTransformer(n_quantiles=min(1000, n_samples), output_distribution='uniform', random_state=seed)
        elif scaler_type == 'quantile_gaussian': return QuantileTransformer(n_quantiles=min(1000, n_samples), output_distribution='normal', random_state=seed)
        elif scaler_type == 'standard': return StandardScaler()
        return RobustScaler()

    def _consolidate(self, cfg, scores, sec):
        scores_arr = np.array(scores, dtype=float)
        if np.all(np.isnan(scores_arr)):
            p_center, p_spread = np.nan, np.nan
        else:
            p_center, p_spread = np.nanmedian(scores_arr), np.nanstd(scores_arr)
        res = {'params': json.dumps(cfg, sort_keys=True, cls=NumpyEncoder), 'primary_center': p_center, 'primary_spread': p_spread}
        if sec and isinstance(sec, list) and len(sec) > 0 and isinstance(sec[0], dict):
            for m in sec[0].keys():
                vals = np.array([d[m] for d in sec], dtype=float)
                if np.all(np.isnan(vals)):
                    if 'pval' in m: res[f'{m}_pass_rate'] = np.nan
                    res[f'{m}_center'], res[f'{m}_spread'] = np.nan, np.nan
                else:
                    if 'pval' in m: res[f'{m}_pass_rate'] = np.nanmean(vals > 0.05)
                    res[f'{m}_center'] = np.nanmedian(vals)
                    res[f'{m}_spread'] = np.nanstd(vals)
        return res