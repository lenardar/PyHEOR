"""
Excel verification export for PyHEOR models.

Exports complete model calculations to Excel for transparent verification.
Each sheet contains one aspect of the model, with formulas and formatting
to facilitate manual checking.

Supports both Markov and PSM model exports.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union


def export_to_excel(
    result,
    filepath: str,
    include_formulas: bool = True,
    include_psa: bool = False,
):
    """Export model results to Excel for verification.

    Creates a multi-sheet Excel workbook with:
    - Summary: Key results, ICER
    - State Trace: State occupancy per cycle
    - Costs: Per-cycle costs by category
    - QALYs: Per-cycle QALYs and LYs
    - Discounting: Discount factors and intermediate calculations
    - Parameters: All parameter values used

    For PSM models, also includes:
    - Survival Curves: Endpoint survival values at each time point

    Parameters
    ----------
    result : BaseResult, PSMBaseResult, OWSAResult, or PSAResult
        Model result to export.
    filepath : str
        Output file path (should end with .xlsx).
    include_formulas : bool
        Not implemented yet (reserved for future Excel formula generation).
    include_psa : bool
        For PSAResult, whether to include all simulation data.
    """
    # Detect result type
    from .results import BaseResult, OWSAResult, PSAResult, PSMBaseResult

    if isinstance(result, PSAResult):
        _export_psa(result, filepath, include_psa)
    elif isinstance(result, OWSAResult):
        _export_owsa(result, filepath)
    elif isinstance(result, PSMBaseResult):
        _export_psm_base(result, filepath)
    elif isinstance(result, BaseResult):
        _export_markov_base(result, filepath)
    else:
        raise TypeError(f"Unsupported result type: {type(result)}")


def _export_markov_base(result, filepath: str):
    """Export Markov base case results to Excel."""
    model = result.model
    params = result.params
    r = result.results

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # === Sheet 1: Parameters ===
        param_rows = []
        for name, val in params.items():
            p = model.params.get(name)
            param_rows.append({
                'Parameter': name,
                'Label': p.label if p else name,
                'Value': val,
                'Distribution': repr(p.dist) if p and p.dist else 'Fixed',
                'OWSA Low': p.low if p else '',
                'OWSA High': p.high if p else '',
            })
        pd.DataFrame(param_rows).to_excel(writer, sheet_name='Parameters', index=False)

        # === Sheet 2: Model Settings ===
        settings = pd.DataFrame([
            {'Setting': 'States', 'Value': ', '.join(model.states)},
            {'Setting': 'Number of States', 'Value': model.n_states},
            {'Setting': 'Strategies', 'Value': ', '.join(model.strategy_names)},
            {'Setting': 'Number of Cycles', 'Value': model.n_cycles},
            {'Setting': 'Cycle Length (years)', 'Value': model.cycle_length},
            {'Setting': 'Discount Rate (costs)', 'Value': model.dr_costs},
            {'Setting': 'Discount Rate (QALYs)', 'Value': model.dr_qalys},
            {'Setting': 'Half-cycle Correction', 'Value': model.half_cycle_correction or 'None'},
            {'Setting': 'Initial State', 'Value': model.states[model.initial_state_idx]},
        ])
        settings.to_excel(writer, sheet_name='Settings', index=False)

        # === Sheet 3: Summary ===
        summary = result.summary()
        summary.to_excel(writer, sheet_name='Summary', index=False)

        # ICER
        try:
            icer_df = result.icer()
            icer_df.to_excel(writer, sheet_name='Summary',
                             startrow=len(summary) + 3, index=False)
        except Exception:
            pass

        # === Per-strategy sheets ===
        for strategy in model.strategy_names:
            sr = r[strategy]
            label = model.strategy_labels[strategy]
            sheet_name = f'Trace_{label}'[:31]  # Excel sheet name limit

            # --- Markov Trace ---
            trace_df = pd.DataFrame(
                sr['trace'],
                columns=model.states,
            )
            trace_df.insert(0, 'Cycle', np.arange(model.n_cycles + 1))
            trace_df.insert(1, 'Time (yrs)',
                            np.arange(model.n_cycles + 1) * model.cycle_length)

            # Add row sum for verification
            trace_df['Row Sum'] = trace_df[model.states].sum(axis=1)

            trace_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # --- Costs sheet ---
            costs_sheet = f'Costs_{label}'[:31]
            cycles = np.arange(model.n_cycles + 1)
            df_c = 1 / (1 + model.dr_costs) ** (cycles * model.cycle_length)

            costs_data = {'Cycle': cycles}
            costs_data['Time (yrs)'] = cycles * model.cycle_length
            costs_data['Discount Factor'] = df_c

            # HCC weights
            hcc = np.ones(model.n_cycles + 1)
            if model.half_cycle_correction == "trapezoidal":
                hcc[0] = 0.5
                hcc[-1] = 0.5
            costs_data['HCC Weight'] = hcc

            total_discounted = np.zeros(model.n_cycles + 1)
            for cat in sr['costs_by_cycle']:
                raw = sr['costs_by_cycle'][cat]
                hcc_applied = sr['costs_hcc'][cat]
                discounted = sr['discounted_costs'][cat]
                costs_data[f'{cat} (raw)'] = raw
                costs_data[f'{cat} (HCC)'] = hcc_applied
                costs_data[f'{cat} (discounted)'] = discounted
                total_discounted += discounted

            costs_data['Total Discounted Cost'] = total_discounted

            costs_df = pd.DataFrame(costs_data)

            # Add totals row
            totals = {col: costs_df[col].sum() if col not in ['Cycle', 'Time (yrs)', 'Discount Factor', 'HCC Weight']
                       else '' for col in costs_df.columns}
            totals['Cycle'] = 'TOTAL'
            totals_df = pd.DataFrame([totals])
            costs_df = pd.concat([costs_df, totals_df], ignore_index=True)

            costs_df.to_excel(writer, sheet_name=costs_sheet, index=False)

            # --- QALYs sheet ---
            qaly_sheet = f'QALYs_{label}'[:31]
            df_q = 1 / (1 + model.dr_qalys) ** (cycles * model.cycle_length)

            qaly_data = {
                'Cycle': cycles,
                'Time (yrs)': cycles * model.cycle_length,
                'Discount Factor': df_q,
                'HCC Weight': hcc,
                'QALYs (raw)': sr['qalys_by_cycle'],
                'QALYs (HCC)': sr['qalys_hcc'],
                'QALYs (discounted)': sr['discounted_qalys'],
                'LYs (raw)': sr['lys_by_cycle'],
                'LYs (HCC)': sr['lys_hcc'],
                'LYs (discounted)': sr['discounted_lys'],
            }
            qaly_df = pd.DataFrame(qaly_data)

            totals_q = {col: qaly_df[col].sum() if col not in ['Cycle', 'Time (yrs)', 'Discount Factor', 'HCC Weight']
                         else '' for col in qaly_df.columns}
            totals_q['Cycle'] = 'TOTAL'
            totals_q_df = pd.DataFrame([totals_q])
            qaly_df = pd.concat([qaly_df, totals_q_df], ignore_index=True)

            qaly_df.to_excel(writer, sheet_name=qaly_sheet, index=False)

        # === Transition Matrices ===
        _write_transition_matrices(writer, model, params)

    print(f"✅ Excel verification exported to: {filepath}")


def _write_transition_matrices(writer, model, params):
    """Write transition matrices for each strategy."""
    from .utils import _Complement, C, resolve_complement

    rows_all = []
    for strategy in model.strategy_names:
        label = model.strategy_labels[strategy]

        # Get matrix at cycle 0 (base case)
        try:
            P = model._get_transition_matrix(strategy, params, 0)
            for i in range(model.n_states):
                row = {'Strategy': label, 'From': model.states[i]}
                for j in range(model.n_states):
                    row[f'To: {model.states[j]}'] = P[i, j]
                row['Row Sum'] = P[i].sum()
                rows_all.append(row)
        except Exception as e:
            rows_all.append({'Strategy': label, 'From': f'Error: {e}'})

    if rows_all:
        pd.DataFrame(rows_all).to_excel(
            writer, sheet_name='Transition Matrices', index=False
        )


def _export_psm_base(result, filepath: str):
    """Export PSM base case results to Excel."""
    model = result.model
    params = result.params
    r = result.results

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # === Parameters ===
        param_rows = []
        for name, val in params.items():
            p = model.params.get(name)
            param_rows.append({
                'Parameter': name,
                'Label': p.label if p else name,
                'Value': val,
                'Distribution': repr(p.dist) if p and p.dist else 'Fixed',
            })
        pd.DataFrame(param_rows).to_excel(writer, sheet_name='Parameters', index=False)

        # === Settings ===
        settings = pd.DataFrame([
            {'Setting': 'Model Type', 'Value': 'Partitioned Survival Model'},
            {'Setting': 'States', 'Value': ', '.join(model.states)},
            {'Setting': 'Survival Endpoints', 'Value': ', '.join(model.survival_endpoints)},
            {'Setting': 'Strategies', 'Value': ', '.join(model.strategy_names)},
            {'Setting': 'Number of Cycles', 'Value': model.n_cycles},
            {'Setting': 'Cycle Length (years)', 'Value': model.cycle_length},
            {'Setting': 'Discount Rate (costs)', 'Value': model.dr_costs},
            {'Setting': 'Discount Rate (QALYs)', 'Value': model.dr_qalys},
            {'Setting': 'Half-cycle Correction', 'Value': model.half_cycle_correction or 'None'},
        ])
        settings.to_excel(writer, sheet_name='Settings', index=False)

        # === Summary ===
        summary = result.summary()
        summary.to_excel(writer, sheet_name='Summary', index=False)
        try:
            icer_df = result.icer()
            icer_df.to_excel(writer, sheet_name='Summary',
                             startrow=len(summary) + 3, index=False)
        except Exception:
            pass

        # === Per-strategy sheets ===
        for strategy in model.strategy_names:
            sr = r[strategy]
            label = model.strategy_labels[strategy]
            times = sr['times']
            cycles = np.arange(model.n_cycles + 1)

            # --- Survival Curves ---
            surv_sheet = f'Surv_{label}'[:31]
            surv_data = {'Cycle': cycles, 'Time (yrs)': times}
            for endpoint in model.survival_endpoints:
                surv_data[f'S({endpoint})'] = sr['survival_curves'][endpoint]
            surv_df = pd.DataFrame(surv_data)
            surv_df.to_excel(writer, sheet_name=surv_sheet, index=False)

            # --- State Probabilities ---
            trace_sheet = f'States_{label}'[:31]
            trace_df = pd.DataFrame(sr['trace'], columns=model.states)
            trace_df.insert(0, 'Cycle', cycles)
            trace_df.insert(1, 'Time (yrs)', times)

            # Verification columns
            trace_df['Row Sum'] = trace_df[model.states].sum(axis=1)
            # Show derivation
            for j, endpoint in enumerate(model.survival_endpoints):
                trace_df[f'S({endpoint})'] = sr['survival_curves'][endpoint]

            trace_df.to_excel(writer, sheet_name=trace_sheet, index=False)

            # --- Costs ---
            costs_sheet = f'Costs_{label}'[:31]
            df_c = 1 / (1 + model.dr_costs) ** (cycles * model.cycle_length)
            hcc = np.ones(model.n_cycles + 1)
            if model.half_cycle_correction == "trapezoidal":
                hcc[0] = 0.5
                hcc[-1] = 0.5

            costs_data = {
                'Cycle': cycles,
                'Time (yrs)': times,
                'Discount Factor': df_c,
                'HCC Weight': hcc,
            }

            total_discounted = np.zeros(model.n_cycles + 1)
            for cat in sr['costs_by_cycle']:
                raw = sr['costs_by_cycle'][cat]
                hcc_applied = sr['costs_hcc'][cat]
                discounted = sr['discounted_costs'][cat]
                costs_data[f'{cat} (raw)'] = raw
                costs_data[f'{cat} (HCC)'] = hcc_applied
                costs_data[f'{cat} (discounted)'] = discounted
                total_discounted += discounted

            costs_data['Total Discounted Cost'] = total_discounted
            costs_df = pd.DataFrame(costs_data)

            totals = {col: costs_df[col].sum() if col not in ['Cycle', 'Time (yrs)', 'Discount Factor', 'HCC Weight']
                       else '' for col in costs_df.columns}
            totals['Cycle'] = 'TOTAL'
            costs_df = pd.concat([costs_df, pd.DataFrame([totals])], ignore_index=True)
            costs_df.to_excel(writer, sheet_name=costs_sheet, index=False)

            # --- QALYs ---
            qaly_sheet = f'QALYs_{label}'[:31]
            df_q = 1 / (1 + model.dr_qalys) ** (cycles * model.cycle_length)

            qaly_data = {
                'Cycle': cycles,
                'Time (yrs)': times,
                'Discount Factor': df_q,
                'HCC Weight': hcc,
                'QALYs (raw)': sr['qalys_by_cycle'],
                'QALYs (HCC)': sr['qalys_hcc'],
                'QALYs (discounted)': sr['discounted_qalys'],
                'LYs (raw)': sr['lys_by_cycle'],
                'LYs (HCC)': sr['lys_hcc'],
                'LYs (discounted)': sr['discounted_lys'],
            }
            qaly_df = pd.DataFrame(qaly_data)

            totals_q = {col: qaly_df[col].sum() if col not in ['Cycle', 'Time (yrs)', 'Discount Factor', 'HCC Weight']
                         else '' for col in qaly_df.columns}
            totals_q['Cycle'] = 'TOTAL'
            qaly_df = pd.concat([qaly_df, pd.DataFrame([totals_q])], ignore_index=True)
            qaly_df.to_excel(writer, sheet_name=qaly_sheet, index=False)

    print(f"✅ Excel verification exported to: {filepath}")


def _export_owsa(result, filepath: str):
    """Export OWSA results to Excel."""
    model = result.model

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # === Base Case Summary ===
        # Reconstruct base case summary from base_result
        from .results import BaseResult
        base_res_obj = BaseResult(model=model, results=result.base_result,
                                  params=result.base_params)
        summary = base_res_obj.summary()
        summary.to_excel(writer, sheet_name='Base Case', index=False)
        try:
            icer_df = base_res_obj.icer()
            icer_df.to_excel(writer, sheet_name='Base Case',
                             startrow=len(summary) + 3, index=False)
        except Exception:
            pass

        # === OWSA Summary ===
        owsa_summary = result.summary()
        owsa_summary.to_excel(writer, sheet_name='OWSA Summary', index=False)

        # === Detailed OWSA results ===
        detail_rows = []
        for entry in result.owsa_data:
            for strategy in model.strategy_names:
                r = entry['result'][strategy]
                total_cost = sum(r['total_costs'].values())
                detail_rows.append({
                    'Parameter': entry['label'] or entry['param'],
                    'Bound': entry['bound'],
                    'Value': entry['value'],
                    'Strategy': model.strategy_labels[strategy],
                    'Total Cost': total_cost,
                    'QALYs': r['total_qalys'],
                    'LYs': r['total_lys'],
                })
        pd.DataFrame(detail_rows).to_excel(
            writer, sheet_name='OWSA Detail', index=False
        )

    print(f"✅ OWSA Excel exported to: {filepath}")


def _export_psa(result, filepath: str, include_all: bool = False):
    """Export PSA results to Excel."""
    model = result.model

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # === PSA Summary ===
        summary = result.summary()
        summary.to_excel(writer, sheet_name='PSA Summary', index=False)

        # === ICER ===
        try:
            icer_df = result.icer()
            icer_df.to_excel(writer, sheet_name='PSA Summary',
                             startrow=len(summary) + 3, index=False)
        except Exception:
            pass

        # === CE Table (all sims) ===
        if include_all:
            ce = result.ce_table
            ce.to_excel(writer, sheet_name='CE Table', index=False)

        # === CEAC Data ===
        try:
            ceac = result.ceac_data()
            ceac.to_excel(writer, sheet_name='CEAC Data', index=False)
        except Exception:
            pass

        # === Sampled Parameters ===
        if include_all and result.sampled_params:
            params_df = pd.DataFrame(result.sampled_params)
            params_df.insert(0, 'Simulation', np.arange(1, len(result.sampled_params) + 1))
            params_df.to_excel(writer, sheet_name='Sampled Parameters', index=False)

    print(f"✅ PSA Excel exported to: {filepath}")


def export_comparison_excel(
    results: Dict[str, Any],
    filepath: str,
    labels: Optional[Dict[str, str]] = None,
):
    """Export a comparison of multiple model results to Excel.

    Useful for comparing Markov vs PSM, or different model configurations.

    Parameters
    ----------
    results : dict
        Maps model names to result objects.
    filepath : str
        Output file path.
    labels : dict, optional
        Custom labels for each model.
    """
    if labels is None:
        labels = {k: k for k in results}

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        all_summaries = []
        for model_name, result in results.items():
            label = labels.get(model_name, model_name)
            summary = result.summary()
            summary.insert(0, 'Model', label)
            all_summaries.append(summary)

        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_excel(writer, sheet_name='Comparison', index=False)

    print(f"✅ Comparison Excel exported to: {filepath}")
