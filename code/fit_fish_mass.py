"""Fish mass fitting API, suitable for Spyser. No command-line arguments.
Units: length_cm and height_cm in cm, mass_g in grams; one row per fish.
Edit USER SETTINGS at the bottom and run, or import run_analysis().
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

PARAMETERS = {'L_power': 2, 'H_power': 2, 'LH_power': 3,'HL_ratio_power': 2 ,'four_term': 8}

def term_values(p, x):
    L, H = x[:, 0], x[:, 1]
    z = np.column_stack((L, H, L*H, L/H))
    return p[::2] * z**p[1::2]

def predict(model, p, x):
    L, H = x[:, 0], x[:, 1]
    if model == 'L_power': return p[0]*L**p[1]
    if model == 'H_power': return p[0]*H**p[1]
    if model == 'LH_power': return p[0]*L**p[1]*H**p[2]
    if model == 'HL_ratio_power': return p[0] * (H / L)**p[1]
    if model == 'four_term': return term_values(p, x).sum(axis=1)
    raise ValueError('Unknown model: '+str(model))

def fit_model(model, x, y, starts=12, seed=42):
    """Bounded multistart least squares in grams; normalization uses training data only."""
    k = PARAMETERS[model]
    scale, yscale = np.median(x, axis=0), float(np.median(y))
    xs = x/scale
    lower, upper = np.zeros(k), np.full(k, 6.0)
    if model == 'four_term':
        upper[::2], lower[1::2] = 100.0, -4.0
    else:
        upper[0], lower[1:] = 100.0, -4.0
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(starts):
        initial = rng.uniform(0.1, 3.0, k)
        if model == 'four_term': initial[::2] = 0.25
        else: initial[0] = 1.0
        result = least_squares(lambda p: predict(model,p,xs)-y/yscale,
            initial, bounds=(lower,upper), x_scale='jac', max_nfev=3000,
            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        if result.success and np.isfinite(result.fun).all(): fits.append(result)
    if not fits: raise RuntimeError('No converged starts for '+model)
    fits.sort(key=lambda r: float(r.fun@r.fun))
    def physical(p):
        q = p.copy()
        if model == 'L_power': q[0] *= yscale/scale[0]**p[1]
        elif model == 'H_power': q[0] *= yscale/scale[1]**p[1]
        elif model == 'LH_power': q[0] *= yscale/(scale[0]**p[1]*scale[1]**p[2])
        elif model == 'HL_ratio_power':
            ratio_scale = scale[1] / scale[0]
            q[0] *= yscale / ratio_scale**p[1]
        else:
            s = np.array([scale[0],scale[1],scale.prod(),scale[0]/scale[1]])
            q[::2] *= yscale/s**p[1::2]
        return q
    best = fits[0]
    sv = np.linalg.svd(best.jac,compute_uv=False)
    diag = {'parameters':physical(best.x).tolist(),
        'parameter_order':['a1','b1','a2','b2','a3','b3','a4','b4'] if model=='four_term' else
                          (['a','bL','bH'] if model=='LH_power' else ['a','b']),
        'residual_df':int(len(y)-k), 'jacobian_rank':int(np.linalg.matrix_rank(best.jac)),
        'jacobian_condition':float(sv[0]/sv[-1]) if sv[-1]>0 else None,
        'parameters_at_bound':int(np.count_nonzero(best.active_mask)),
        'converged_starts':len(fits)}
    candidates = [{'model':model,'start_rank':j,'SSE_g2':float(r.fun@r.fun*yscale**2),
                   'parameters_json':json.dumps(physical(r.x).tolist())} for j,r in enumerate(fits)]
    return physical(best.x),diag,candidates

def error_metrics(y, prediction):
    e = prediction-y
    sst = np.sum((y-y.mean())**2)
    return {'MAE_g':float(np.mean(abs(e))), 'RMSE_g':float(np.sqrt(np.mean(e**2))),
        'MAPE_percent':float(100*np.mean(abs(e)/y)), 'bias_g':float(np.mean(e)),
        'R2':float(1-np.sum(e**2)/sst) if sst>0 else None}

def make_demo_data(seed=42, n=24):
    """Synthetic demonstration only; not experimental evidence."""
    rng = np.random.default_rng(seed)
    L = rng.uniform(3,8.4,n)
    H = L*rng.uniform(0.25,0.48,n)
    return pd.DataFrame({'fish_id':['demo_%02d'%i for i in range(n)],
        'species':'SYNTHETIC','length_cm':L,'height_cm':H,
        'mass_g':0.08*L**1.6*H**1.1*np.exp(rng.normal(0,0.08,n))})

def run_analysis(data, output_dir='mass_results', models=None, starts=12,
                 seed=42, run_loocv=True, show_plots=True, verbose=True):
    """Fit models and return all outputs as a dictionary.

    data: pandas DataFrame or CSV path. Required columns:
          fish_id, length_cm, height_cm, mass_g. species is optional.
    output_dir: output folder, or None to disable disk writes.
    models: sequence of model names; None selects all five.
    run_loocv: refit each eligible model leaving one fish out at a time.
               Conservatively skipped if training n <= parameter count.
    show_plots: display the figure in Spyser; saving is independent of display.
    Returns: metrics, predictions, diagnostics, multistart_solutions,
             contributions, fitted_parameters, cv_fits, figure, input_data.
    No model is selected automatically. No confidence intervals are inferred.
    """
    if not isinstance(starts,int) or starts<1: raise ValueError('starts must be a positive integer')
    models = list(PARAMETERS) if models is None else list(models)
    if not models or len(models)!=len(set(models)) or any(m not in PARAMETERS for m in models):
        raise ValueError('Use distinct model names from '+str(list(PARAMETERS)))
    df = data.copy() if isinstance(data,pd.DataFrame) else pd.read_csv(data)
    required = ['fish_id','length_cm','height_cm','mass_g']
    if not set(required).issubset(df.columns): raise ValueError('Required: '+', '.join(required))
    if df.fish_id.isna().any() or df.fish_id.duplicated().any():
        raise ValueError('Use one row and one unique fish_id per fish')
    for c in required[1:]: df[c] = pd.to_numeric(df[c],errors='raise')
    x, y = df[['length_cm','height_cm']].to_numpy(float), df.mass_g.to_numpy(float)
    if len(y)<4: raise ValueError('At least four fish required for this workflow')
    if not np.isfinite(np.column_stack((x,y))).all() or (x<=0).any() or (y<=0).any():
        raise ValueError('Dimensions and masses must be finite and strictly positive')
    predictions, rows, diagnostics, candidates = df.copy(), [], {}, []
    params, cv_rows, contributions = {}, [], pd.DataFrame()
    fig, axes = plt.subplots(1,2,figsize=(11,5))
    for model in models:
        k = PARAMETERS[model]
        if len(y)<=k:
            diagnostics[model] = {'status':'skipped: n <= parameter count'}
            continue
        if verbose: print('Fitting',model)
        p, diag, trials = fit_model(model,x,y,starts,seed)
        params[model], diagnostics[model] = p, diag
        candidates.extend(trials)
        fitted = predict(model,p,x)
        predictions[model+'_fitted_g'] = fitted
        rows.append({'model':model,'evaluation':'resubstitution','n':len(y),'parameters':k,**error_metrics(y,fitted)})
        axes[0].scatter(y,fitted,label=model,alpha=0.75)
        if run_loocv and len(y)-1>k:
            loo = np.empty(len(y))
            for i in range(len(y)):
                keep = np.arange(len(y))!=i
                q, fold_diag, _ = fit_model(model,x[keep],y[keep],starts,seed+i+1)
                loo[i] = predict(model,q,x[i:i+1])[0]
                cv_rows.append({'model':model,'held_out_fish_id':str(df.iloc[i].fish_id),
                    'prediction_g':float(loo[i]),'parameters_json':json.dumps(q.tolist()),
                    'jacobian_rank':fold_diag['jacobian_rank'],
                    'parameters_at_bound':fold_diag['parameters_at_bound']})
            predictions[model+'_LOO_g'] = loo
            rows.append({'model':model,'evaluation':'leave_one_fish_out','n':len(y),'parameters':k,**error_metrics(y,loo)})
            axes[1].scatter(y,loo,label=model,alpha=0.75)
            diag['LOO_status'] = 'completed; exploratory internal validation'
        else:
            diag['LOO_status'] = 'disabled' if not run_loocv else 'skipped: training n <= parameter count'
        if model=='four_term':
            t = term_values(p,x)
            contributions = pd.DataFrame(t,columns=['L_term_g','H_term_g','LH_term_g','ratio_term_g'])
            contributions.insert(0,'fish_id',df.fish_id.to_numpy())
            for j,name in enumerate(['L','H','LH','ratio']): contributions[name+'_share'] = t[:,j]/fitted
    for ax,title in zip(axes,['All-fish fitted values','Leave-one-fish-out predictions']):
        ax.plot([0,y.max()*1.1],[0,y.max()*1.1],'k--',linewidth=1)
        ax.set(xlabel='Reference mass (g)',ylabel='Predicted mass (g)',title=title)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    fig.tight_layout()
    result = {'metrics':pd.DataFrame(rows),'predictions':predictions,'diagnostics':diagnostics,
        'multistart_solutions':pd.DataFrame(candidates),'contributions':contributions,
        'fitted_parameters':params,'cv_fits':pd.DataFrame(cv_rows),'figure':fig,'input_data':df}
    if output_dir is not None:
        out = Path(output_dir).expanduser(); out.mkdir(parents=True,exist_ok=True)
        for key,filename in [('metrics','model_metrics.csv'),('predictions','fish_predictions.csv'),
            ('multistart_solutions','multistart_solutions.csv'),('contributions','four_term_contributions.csv'),
            ('cv_fits','loocv_fits.csv'),('input_data','input_used.csv')]:
            if not result[key].empty: result[key].to_csv(out/filename,index=False)
        (out/'fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False))
        fig.savefig(out/'mass_predictions.png',dpi=160)
    if verbose:
        print(result['metrics'].to_string(index=False))
        print('Inspect diagnostics: low rank, high conditioning, and active bounds indicate instability.')
        print('Term shares are fitted contributions, not biological importance.')
    if show_plots: plt.show()
    else: plt.close(fig)
    return result

# USER SETTINGS: edit these, then press Run in Spyser.
if __name__ == '__main__':
    USE_DEMO = False  # Switch to False for your measured data.
    BASE_DIR = Path(__file__).resolve().parent
    CSV_PATH = BASE_DIR / 'fish_mass_info.csv'
    OUTPUT_DIR = BASE_DIR / ('mass_demo_results' if USE_DEMO else 'mass_results')
    N_STARTS = 12
    RUN_LOOCV = True
    SHOW_PLOTS = True

    fish_data = make_demo_data() if USE_DEMO else pd.read_csv(CSV_PATH)
    results = run_analysis(fish_data, output_dir=OUTPUT_DIR, starts=N_STARTS,
                           run_loocv=RUN_LOOCV, show_plots=SHOW_PLOTS)
    metrics = results['metrics']
    predictions = results['predictions']
    diagnostics = results['diagnostics']
    fitted_parameters = results['fitted_parameters']
    contributions = results['contributions']
    cv_fits = results['cv_fits']
