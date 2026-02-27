"""
Plotting module for PyHEOR.

Provides beautiful, publication-ready visualizations:
- State transition diagrams
- TreeAge-style model structure diagrams
- Markov trace plots
- Tornado diagrams (OWSA)
- Cost-effectiveness planes (PSA scatter)
- Cost-effectiveness acceptability curves (CEAC)
- PSA convergence plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, RegularPolygon
from matplotlib.path import Path
import matplotlib.patheffects as pe
from typing import Optional, List, Dict, Tuple

# =============================================================================
# Color Palette & Style
# =============================================================================

COLORS = {
    'strategies': ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800', '#607D8B'],
    'states': ['#E8F5E9', '#BBDEFB', '#FFF9C4', '#FFCCBC', '#E1BEE7', '#B3E5FC'],
    'state_border': ['#4CAF50', '#1976D2', '#FBC02D', '#E64A19', '#7B1FA2', '#0288D1'],
    'state_dark': ['#2E7D32', '#0D47A1', '#F57F17', '#BF360C', '#4A148C', '#01579B'],
    'positive': '#4CAF50',
    'negative': '#FF5722',
    'neutral': '#9E9E9E',
    'wtp_line': '#E91E63',
    'grid': '#E0E0E0',
    'bg': '#FAFAFA',
}


def _setup_style():
    """Apply clean plot styling."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.color': COLORS['grid'],
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#E0E0E0',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _get_strategy_colors(n: int) -> list:
    """Get n strategy colors."""
    base = COLORS['strategies']
    if n <= len(base):
        return base[:n]
    # Extend with repeats
    return (base * ((n // len(base)) + 1))[:n]


def _get_state_colors(n: int) -> Tuple[list, list]:
    """Get n state fill and border colors."""
    fills = COLORS['states']
    borders = COLORS['state_border']
    if n <= len(fills):
        return fills[:n], borders[:n]
    fills = (fills * ((n // len(fills)) + 1))[:n]
    borders = (borders * ((n // len(borders)) + 1))[:n]
    return fills, borders


# =============================================================================
# State Transition Diagram
# =============================================================================

def _compute_layout(n_states: int, states: list) -> list:
    """Compute node positions for transition diagram.
    
    Uses a smart layout:
    - 2 states: horizontal
    - 3 states: triangle (2 top, 1 bottom) or inverted
    - 4+ states: circular with absorbing states at bottom
    """
    if n_states == 2:
        return [(-1.5, 0), (1.5, 0)]
    elif n_states == 3:
        # Common pattern: 2 alive + 1 dead
        return [(-1.8, 1.0), (1.8, 1.0), (0, -1.5)]
    elif n_states == 4:
        return [(-2, 1.2), (2, 1.2), (-2, -1.2), (2, -1.2)]
    elif n_states == 5:
        return [(-2, 1.5), (2, 1.5), (-2.5, -0.3), (2.5, -0.3), (0, -2)]
    else:
        angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, n_states, endpoint=False)
        radius = 2.0
        return [(radius * np.cos(a), radius * np.sin(a)) for a in angles]


def _edge_point(center: tuple, target: tuple, radius: float) -> tuple:
    """Compute point on circle edge facing toward target."""
    dx = target[0] - center[0]
    dy = target[1] - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-10:
        return center
    return (center[0] + radius * dx / dist, center[1] + radius * dy / dist)


def plot_transition_diagram(
    model, params: dict, strategy: Optional[str] = None,
    cycle: int = 1, figsize: tuple = (10, 8),
    node_radius: float = 0.55, show_probs: bool = True,
    min_prob: float = 0.001, title: Optional[str] = None,
    ax=None,
):
    """Plot state transition diagram.
    
    Parameters
    ----------
    model : MarkovModel
        The model.
    params : dict
        Parameter values.
    strategy : str, optional
        Strategy to show (default: first strategy).
    cycle : int
        Cycle at which to evaluate transition probabilities.
    figsize : tuple
        Figure size.
    node_radius : float
        Radius of state nodes.
    show_probs : bool
        Whether to show transition probabilities on arrows.
    min_prob : float
        Minimum probability to show an arrow.
    title : str, optional
        Custom title.
    """
    _setup_style()
    
    if strategy is None:
        strategy = model.strategy_names[0]
    
    P = model._get_transition_matrix(strategy, params, cycle)
    n = model.n_states
    states = model.states
    
    positions = _compute_layout(n, states)
    fills, borders = _get_state_colors(n)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    ax.set_facecolor('white')
    
    # --- Draw edges (transitions between different states) ---
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if P[i, j] < min_prob:
                continue
            
            pos_i = positions[i]
            pos_j = positions[j]
            
            # Start and end on circle edges
            start = _edge_point(pos_i, pos_j, node_radius + 0.02)
            end = _edge_point(pos_j, pos_i, node_radius + 0.02)
            
            # Determine curvature (offset if bidirectional)
            has_reverse = P[j, i] >= min_prob
            rad = 0.25 if has_reverse else 0.15
            
            # Arrow color based on probability
            alpha = max(0.3, min(1.0, P[i, j] * 3))
            color = borders[j]
            
            arrow = FancyArrowPatch(
                posA=start, posB=end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle='-|>',
                mutation_scale=18,
                linewidth=1.5 + P[i, j] * 3,
                color=color,
                alpha=alpha,
                zorder=2,
            )
            ax.add_patch(arrow)
            
            # Label with probability
            if show_probs:
                # Position label along the arrow
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                # Offset for curvature
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                offset_x = -dy * rad * 0.4
                offset_y = dx * rad * 0.4
                
                label_x = mid_x + offset_x
                label_y = mid_y + offset_y
                
                ax.text(
                    label_x, label_y,
                    f'{P[i,j]:.3f}',
                    fontsize=8, ha='center', va='center',
                    color=color,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             edgecolor='none', alpha=0.85),
                    zorder=5,
                )
    
    # --- Draw self-loops ---
    for i in range(n):
        if P[i, i] < min_prob:
            continue
        
        x, y = positions[i]
        
        # Find direction away from center for loop placement
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        dx = x - cx
        dy = y - cy
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.1:
            dx, dy = 0, 1  # Default: loop goes up
            dist = 1
        
        # Normalize direction
        nx, ny = dx / dist, dy / dist
        
        # Two points on the node circumference, spread around the outward direction
        angle = np.arctan2(ny, nx)
        spread = 0.4
        
        theta1 = angle - spread
        theta2 = angle + spread
        
        p1 = (x + node_radius * np.cos(theta1), y + node_radius * np.sin(theta1))
        p2 = (x + node_radius * np.cos(theta2), y + node_radius * np.sin(theta2))
        
        # Use a very curved connection for the self-loop
        arrow = FancyArrowPatch(
            posA=p1, posB=p2,
            connectionstyle=f"arc3,rad=-2.0",
            arrowstyle='-|>',
            mutation_scale=14,
            linewidth=1.2 + P[i, i] * 2,
            color=borders[i],
            alpha=0.6,
            zorder=2,
        )
        ax.add_patch(arrow)
        
        if show_probs:
            lx = x + nx * (node_radius + 0.55)
            ly = y + ny * (node_radius + 0.55)
            ax.text(
                lx, ly,
                f'{P[i,i]:.3f}',
                fontsize=8, ha='center', va='center',
                color=borders[i],
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='none', alpha=0.85),
                zorder=5,
            )
    
    # --- Draw nodes ---
    for i, (x, y) in enumerate(positions):
        circle = plt.Circle(
            (x, y), node_radius,
            facecolor=fills[i],
            edgecolor=borders[i],
            linewidth=2.5,
            zorder=10,
        )
        ax.add_patch(circle)
        
        ax.text(
            x, y, states[i],
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color=COLORS['state_dark'][i % len(COLORS['state_dark'])],
            zorder=11,
        )
    
    # --- Formatting ---
    all_x = [p[0] for p in positions]
    all_y = [p[1] for p in positions]
    margin = node_radius + 1.2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title is None:
        title = f'State Transition Diagram — {model.strategy_labels[strategy]}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    fig.tight_layout()
    return fig


# =============================================================================
# Model Structure Diagram (TreeAge Style)
# =============================================================================

def plot_model_diagram(model, figsize: tuple = (14, 7), title: Optional[str] = None):
    """Plot TreeAge-style model structure diagram.
    
    Shows decision node → strategy branches → Markov nodes → health states.
    
    Parameters
    ----------
    model : MarkovModel
        The model.
    figsize : tuple
        Figure size.
    title : str, optional
        Custom title.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')
    
    n_strategies = model.n_strategies
    n_states = model.n_states
    
    # Layout parameters
    x_decision = 1.0
    x_markov = 4.5
    x_states = 8.0
    x_terminal = 10.5
    
    # Total height needed
    state_spacing = 0.8
    strategy_spacing = n_states * state_spacing + 1.5
    total_height = n_strategies * strategy_spacing
    
    y_start = total_height / 2
    
    # --- Decision node (square) ---
    decision_y = 0
    decision_size = 0.35
    
    decision = FancyBboxPatch(
        (x_decision - decision_size, decision_y - decision_size),
        decision_size * 2, decision_size * 2,
        boxstyle="square,pad=0",
        facecolor='#E3F2FD', edgecolor='#1565C0',
        linewidth=2.5, zorder=10,
    )
    ax.add_patch(decision)
    ax.text(x_decision, decision_y, 'D', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#1565C0', zorder=11)
    
    strategy_colors = _get_strategy_colors(n_strategies)
    
    for s_idx, strategy in enumerate(model.strategy_names):
        # Y position for this strategy branch
        y_strategy = y_start - s_idx * strategy_spacing - strategy_spacing / 2
        
        # --- Branch from decision to Markov node ---
        ax.plot(
            [x_decision + decision_size, x_markov - 0.4],
            [decision_y, y_strategy],
            color=strategy_colors[s_idx], linewidth=2,
            solid_capstyle='round', zorder=3,
        )
        
        # Strategy label on branch
        mid_x = (x_decision + decision_size + x_markov - 0.4) / 2
        mid_y = (decision_y + y_strategy) / 2
        ax.text(
            mid_x, mid_y + 0.2,
            model.strategy_labels[strategy],
            fontsize=10, fontweight='bold',
            color=strategy_colors[s_idx],
            ha='center', va='bottom',
            rotation=0,
        )
        
        # --- Markov node (circle with M) ---
        markov_circle = plt.Circle(
            (x_markov, y_strategy), 0.35,
            facecolor='#E8F5E9', edgecolor='#2E7D32',
            linewidth=2.5, zorder=10,
        )
        ax.add_patch(markov_circle)
        ax.text(x_markov, y_strategy, 'M', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#2E7D32', zorder=11)
        
        # --- State branches ---
        fills, borders = _get_state_colors(n_states)
        
        for st_idx, state in enumerate(model.states):
            y_state = y_strategy + (st_idx - (n_states - 1) / 2) * state_spacing
            
            # Branch line
            ax.plot(
                [x_markov + 0.35, x_states - 0.6],
                [y_strategy, y_state],
                color='#666666', linewidth=1.5,
                solid_capstyle='round', zorder=3,
            )
            
            # State box
            box_w = 1.2
            box_h = 0.35
            state_box = FancyBboxPatch(
                (x_states - box_w/2, y_state - box_h),
                box_w, box_h * 2,
                boxstyle="round,pad=0.1",
                facecolor=fills[st_idx],
                edgecolor=borders[st_idx],
                linewidth=2, zorder=10,
            )
            ax.add_patch(state_box)
            ax.text(
                x_states, y_state, state,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color=COLORS['state_dark'][st_idx % len(COLORS['state_dark'])],
                zorder=11,
            )
            
            # Terminal line
            ax.plot(
                [x_states + box_w/2, x_terminal - 0.15],
                [y_state, y_state],
                color='#999999', linewidth=1,
                solid_capstyle='round', zorder=3,
            )
            
            # Terminal node (small triangle)
            triangle = RegularPolygon(
                (x_terminal, y_state), 3,
                radius=0.15, orientation=np.pi/6,
                facecolor=fills[st_idx],
                edgecolor=borders[st_idx],
                linewidth=1.5, zorder=10,
            )
            ax.add_patch(triangle)
    
    # --- Formatting ---
    ax.set_xlim(0, x_terminal + 1)
    y_positions = []
    for s_idx in range(n_strategies):
        yc = y_start - s_idx * strategy_spacing - strategy_spacing / 2
        for st_idx in range(n_states):
            y_positions.append(yc + (st_idx - (n_states - 1) / 2) * state_spacing)
    
    y_margin = 1.0
    ax.set_ylim(min(y_positions) - y_margin, max(y_positions) + y_margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title is None:
        title = 'Model Structure'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    fig.tight_layout()
    return fig


# =============================================================================
# Markov Trace Plot
# =============================================================================

def plot_trace(
    result, style: str = "area", figsize: tuple = (12, 5),
    title: Optional[str] = None, per_strategy: bool = True,
):
    """Plot Markov trace (state occupancy over time).
    
    Parameters
    ----------
    result : BaseResult
        Base case result.
    style : str
        "area" for stacked area chart, "line" for line chart.
    figsize : tuple
        Figure size.
    title : str, optional
        Custom title.
    per_strategy : bool
        If True, creates separate subplots per strategy.
    """
    _setup_style()
    
    model = result.model
    strategies = model.strategy_names
    states = model.states
    fills, borders = _get_state_colors(model.n_states)
    
    n_strat = len(strategies)
    
    if per_strategy and n_strat > 1:
        fig, axes = plt.subplots(1, n_strat, figsize=figsize, sharey=True)
        if n_strat == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
        strategies = [strategies[0]]  # Only plot first if not per_strategy
    
    for idx, (ax, strategy) in enumerate(zip(axes, strategies)):
        trace = result.results[strategy]['trace']
        cycles = np.arange(model.n_cycles + 1) * model.cycle_length
        
        if style == "area":
            ax.stackplot(
                cycles, trace.T,
                labels=states if idx == 0 else ['']*len(states),
                colors=fills,
                alpha=0.85,
                edgecolor='white',
                linewidth=0.5,
            )
        else:
            for s_idx, state in enumerate(states):
                ax.plot(
                    cycles, trace[:, s_idx],
                    label=state if idx == 0 else '',
                    color=borders[s_idx],
                    linewidth=2,
                )
        
        ax.set_xlabel('Time (years)')
        if idx == 0:
            ax.set_ylabel('State Occupancy')
        ax.set_title(model.strategy_labels[strategy], fontsize=12)
        ax.set_xlim(0, cycles[-1])
        ax.set_ylim(0, 1.0)
    
    # Legend
    if per_strategy:
        handles = [mpatches.Patch(facecolor=fills[i], edgecolor=borders[i], 
                                   label=states[i]) for i in range(len(states))]
        fig.legend(handles=handles, loc='lower center', ncol=len(states),
                   bbox_to_anchor=(0.5, -0.02), fontsize=10)
    else:
        axes[0].legend(loc='upper right')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    fig.tight_layout()
    return fig


# =============================================================================
# Tornado Diagram (OWSA)
# =============================================================================

def plot_tornado(
    owsa_result, comparator=None, outcome="nmb",
    figsize: tuple = (10, None), title: Optional[str] = None,
    max_params: int = 15,
):
    """Plot tornado diagram for one-way sensitivity analysis.
    
    Parameters
    ----------
    owsa_result : OWSAResult
        OWSA result object.
    comparator : str, optional
        Comparator strategy.
    outcome : str
        "nmb" for net monetary benefit.
    figsize : tuple
        Figure size (height auto-calculated if None).
    title : str, optional
        Custom title.
    max_params : int
        Maximum number of parameters to show.
    """
    _setup_style()
    
    summary = owsa_result.summary(comparator=comparator, outcome=outcome)
    summary = summary.head(max_params)
    
    n_params = len(summary)
    if figsize[1] is None:
        figsize = (figsize[0], max(4, n_params * 0.5 + 1.5))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    base_inmb = summary['INMB (Base)'].iloc[0]
    
    y_pos = np.arange(n_params)
    
    for i, (_, row) in enumerate(summary.iterrows()):
        low_inmb = row['INMB (Low)']
        high_inmb = row['INMB (High)']
        
        # Range relative to base
        left = min(low_inmb, high_inmb)
        right = max(low_inmb, high_inmb)
        
        # Color: green for positive impact, red for negative
        color_left = COLORS['negative'] if left < base_inmb else COLORS['positive']
        color_right = COLORS['positive'] if right > base_inmb else COLORS['negative']
        
        # Draw two bars: left of base and right of base
        if left < base_inmb:
            ax.barh(n_params - 1 - i, base_inmb - left, left=left,
                    height=0.6, color=COLORS['negative'], alpha=0.8,
                    edgecolor='white', linewidth=0.5)
        if right > base_inmb:
            ax.barh(n_params - 1 - i, right - base_inmb, left=base_inmb,
                    height=0.6, color=COLORS['positive'], alpha=0.8,
                    edgecolor='white', linewidth=0.5)
        if left >= base_inmb:
            ax.barh(n_params - 1 - i, right - left, left=left,
                    height=0.6, color=COLORS['positive'], alpha=0.8,
                    edgecolor='white', linewidth=0.5)
        if right <= base_inmb:
            ax.barh(n_params - 1 - i, right - left, left=left,
                    height=0.6, color=COLORS['negative'], alpha=0.8,
                    edgecolor='white', linewidth=0.5)
        
        # Annotations: low and high bounds
        ax.text(left - abs(right - left) * 0.02, n_params - 1 - i,
                f'{row["Low Value"]:.3g}', ha='right', va='center',
                fontsize=8, color='#666')
        ax.text(right + abs(right - left) * 0.02, n_params - 1 - i,
                f'{row["High Value"]:.3g}', ha='left', va='center',
                fontsize=8, color='#666')
    
    # Base case line
    ax.axvline(base_inmb, color='#333', linewidth=1.5, linestyle='-', zorder=5)
    
    ax.set_yticks(np.arange(n_params))
    ax.set_yticklabels(summary['Parameter'].values[::-1], fontsize=10)
    ax.set_xlabel('Incremental Net Monetary Benefit ($)', fontsize=11)
    
    if title is None:
        title = f'Tornado Diagram (WTP = ${owsa_result.wtp:,.0f}/QALY)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3)
    ax.grid(axis='y', visible=False)
    
    fig.tight_layout()
    return fig


# =============================================================================
# OWSA Single Parameter Plot
# =============================================================================

def plot_owsa_param(
    owsa_result, param_name: str, comparator=None,
    figsize: tuple = (8, 5), title: Optional[str] = None,
):
    """Plot one-way sensitivity for a specific parameter.
    
    Shows how outcomes change as a single parameter varies.
    """
    _setup_style()
    
    if comparator is None:
        comparator = owsa_result.model.strategy_names[0]
    
    intervention = [s for s in owsa_result.model.strategy_names if s != comparator][0]
    
    entries = [d for d in owsa_result.owsa_data if d['param'] == param_name]
    if not entries:
        raise ValueError(f"Parameter '{param_name}' not found in OWSA results")
    
    base_val = entries[0]['base_value']
    
    x_vals = [base_val]
    inmb_vals = []
    
    # Base case INMB
    base = owsa_result.base_result
    base_ic = sum(base[intervention]['total_costs'].values()) - sum(base[comparator]['total_costs'].values())
    base_iq = base[intervention]['total_qalys'] - base[comparator]['total_qalys']
    base_inmb = base_iq * owsa_result.wtp - base_ic
    inmb_vals.append(base_inmb)
    
    for entry in entries:
        x_vals.append(entry['value'])
        r = entry['result']
        ic = sum(r[intervention]['total_costs'].values()) - sum(r[comparator]['total_costs'].values())
        iq = r[intervention]['total_qalys'] - r[comparator]['total_qalys']
        inmb_vals.append(iq * owsa_result.wtp - ic)
    
    # Sort by x
    order = np.argsort(x_vals)
    x_vals = np.array(x_vals)[order]
    inmb_vals = np.array(inmb_vals)[order]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x_vals, inmb_vals, 'o-', color=COLORS['strategies'][0],
            linewidth=2, markersize=8, zorder=5)
    ax.axhline(0, color='#999', linewidth=1, linestyle='--')
    ax.axvline(base_val, color='#999', linewidth=1, linestyle=':', alpha=0.5)
    
    # Mark base case
    ax.plot(base_val, base_inmb, 's', color=COLORS['strategies'][1],
            markersize=10, zorder=6, label='Base case')
    
    param_label = entries[0].get('label', param_name)
    ax.set_xlabel(param_label, fontsize=11)
    ax.set_ylabel('Incremental NMB ($)', fontsize=11)
    
    if title is None:
        title = f'One-Way Sensitivity: {param_label}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    fig.tight_layout()
    return fig


# =============================================================================
# CE Scatter Plot (PSA)
# =============================================================================

def plot_scatter(
    psa_result, comparator=None, wtp: Optional[float] = None,
    figsize: tuple = (9, 7), title: Optional[str] = None,
    alpha: float = 0.3,
):
    """Plot cost-effectiveness scatter (incremental CE plane).
    
    Parameters
    ----------
    psa_result : PSAResult
        PSA result object.
    comparator : str, optional
        Comparator strategy.
    wtp : float, optional
        WTP threshold to show as a line.
    figsize : tuple
        Figure size.
    title : str, optional
        Custom title.
    alpha : float
        Point transparency.
    """
    _setup_style()
    
    if comparator is None:
        comparator = psa_result.model.strategy_names[0]
    
    ce = psa_result.ce_table
    comp_df = ce[ce['strategy'] == comparator].sort_values('sim')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    strategies = [s for s in psa_result.model.strategy_names if s != comparator]
    colors = _get_strategy_colors(len(strategies))
    
    for idx, strategy in enumerate(strategies):
        int_df = ce[ce['strategy'] == strategy].sort_values('sim')
        
        inc_qaly = int_df['qalys'].values - comp_df['qalys'].values
        inc_cost = int_df['total_cost'].values - comp_df['total_cost'].values
        
        ax.scatter(
            inc_qaly, inc_cost,
            s=15, alpha=alpha,
            color=colors[idx],
            label=psa_result.model.strategy_labels[strategy],
            edgecolors='none',
        )
        
        # Mean point
        ax.scatter(
            [inc_qaly.mean()], [inc_cost.mean()],
            s=150, marker='*',
            color=colors[idx], edgecolors='black', linewidth=1,
            zorder=10,
        )
    
    # Axes
    ax.axhline(0, color='#999', linewidth=0.8)
    ax.axvline(0, color='#999', linewidth=0.8)
    
    # WTP line
    if wtp is not None:
        xlims = ax.get_xlim()
        x_range = np.linspace(xlims[0], xlims[1], 100)
        ax.plot(x_range, x_range * wtp, '--',
                color=COLORS['wtp_line'], linewidth=1.5,
                label=f'WTP = ${wtp:,.0f}/QALY')
    
    # Quadrant labels
    ax.text(0.98, 0.98, 'NE\n(More costly,\nmore effective)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='#999', alpha=0.6)
    ax.text(0.02, 0.02, 'SW\n(Less costly,\nless effective)',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, color='#999', alpha=0.6)
    ax.text(0.02, 0.98, 'NW\n(More costly,\nless effective)',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8, color='#999', alpha=0.6)
    ax.text(0.98, 0.02, 'SE\n(Less costly,\nmore effective)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#999', alpha=0.6)
    
    ax.set_xlabel('Incremental QALYs', fontsize=12)
    ax.set_ylabel('Incremental Cost ($)', fontsize=12)
    
    if title is None:
        title = 'Cost-Effectiveness Plane'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    
    fig.tight_layout()
    return fig


# =============================================================================
# CEAC Plot
# =============================================================================

def plot_ceac(
    psa_result, comparator=None,
    wtp_range: tuple = (0, 100000), n_wtp: int = 200,
    figsize: tuple = (10, 6), title: Optional[str] = None,
):
    """Plot cost-effectiveness acceptability curve (CEAC).
    
    Parameters
    ----------
    psa_result : PSAResult
        PSA result object.
    comparator : str, optional
        Comparator strategy.
    wtp_range : tuple
        (min, max) WTP range.
    n_wtp : int
        Number of WTP points.
    figsize : tuple
        Figure size.
    title : str, optional
        Custom title.
    """
    _setup_style()
    
    ceac = psa_result.ceac_data(comparator=comparator, wtp_range=wtp_range, n_wtp=n_wtp)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    strategies = ceac['strategy'].unique()
    colors = _get_strategy_colors(len(strategies))
    
    for idx, strategy in enumerate(strategies):
        df_s = ceac[ceac['strategy'] == strategy]
        label = df_s['Strategy'].iloc[0]
        ax.plot(
            df_s['WTP'], df_s['Prob CE'],
            color=colors[idx], linewidth=2.5,
            label=label,
        )
    
    ax.set_xlabel('Willingness-to-Pay ($/QALY)', fontsize=12)
    ax.set_ylabel('Probability Cost-Effective', fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(wtp_range)
    
    # Reference line at 0.5
    ax.axhline(0.5, color='#999', linewidth=0.8, linestyle=':', alpha=0.5)
    
    if title is None:
        title = 'Cost-Effectiveness Acceptability Curve'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    
    fig.tight_layout()
    return fig


# =============================================================================
# PSA Convergence Plot
# =============================================================================

def plot_convergence(
    psa_result, comparator=None, wtp: float = 50000,
    figsize: tuple = (10, 5), title: Optional[str] = None,
):
    """Plot PSA convergence (running mean of incremental NMB).
    
    Parameters
    ----------
    psa_result : PSAResult
        PSA result object.
    comparator : str, optional
        Comparator strategy.
    wtp : float
        WTP for NMB calculation.
    figsize : tuple
        Figure size.
    title : str, optional
        Custom title.
    """
    _setup_style()
    
    if comparator is None:
        comparator = psa_result.model.strategy_names[0]
    
    ce = psa_result.ce_table
    comp_df = ce[ce['strategy'] == comparator].sort_values('sim')
    
    strategies = [s for s in psa_result.model.strategy_names if s != comparator]
    colors = _get_strategy_colors(len(strategies))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for idx, strategy in enumerate(strategies):
        int_df = ce[ce['strategy'] == strategy].sort_values('sim')
        
        inc_qaly = int_df['qalys'].values - comp_df['qalys'].values
        inc_cost = int_df['total_cost'].values - comp_df['total_cost'].values
        inmb = inc_qaly * wtp - inc_cost
        
        # Running mean
        running_mean = np.cumsum(inmb) / np.arange(1, len(inmb) + 1)
        
        ax.plot(
            np.arange(1, len(inmb) + 1), running_mean,
            color=colors[idx], linewidth=2,
            label=psa_result.model.strategy_labels[strategy],
        )
    
    ax.axhline(0, color='#999', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Number of Simulations', fontsize=12)
    ax.set_ylabel('Running Mean INMB ($)', fontsize=12)
    
    if title is None:
        title = f'PSA Convergence (WTP = ${wtp:,.0f}/QALY)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    fig.tight_layout()
    return fig


# =============================================================================
# PSM-Specific Plots
# =============================================================================

def plot_survival_curves(
    psm_result, figsize: tuple = (10, 6), title: Optional[str] = None,
    endpoints: Optional[List[str]] = None,
    show_legend: bool = True,
):
    """Plot survival curves from a PSM result.

    Shows overlaid survival curves for each strategy and endpoint.
    """
    _setup_style()

    model = psm_result.model
    if endpoints is None:
        endpoints = model.survival_endpoints

    n_strategies = model.n_strategies
    colors = _get_strategy_colors(n_strategies)

    line_styles = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(figsize=figsize)

    for s_idx, strategy in enumerate(model.strategy_names):
        r = psm_result.results[strategy]
        times = r['times']
        for e_idx, endpoint in enumerate(endpoints):
            surv = r['survival_curves'][endpoint]
            ls = line_styles[e_idx % len(line_styles)]
            label = f"{model.strategy_labels[strategy]} — {endpoint}"
            ax.plot(times, surv, color=colors[s_idx], linestyle=ls,
                    linewidth=2, label=label)

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)

    if title is None:
        title = 'Survival Curves'
    ax.set_title(title, fontsize=14, fontweight='bold')

    if show_legend:
        ax.legend(loc='best', fontsize=9)

    fig.tight_layout()
    return fig


def plot_state_area(
    psm_result, strategy: Optional[str] = None,
    figsize: tuple = (10, 6), title: Optional[str] = None,
    alpha: float = 0.7,
):
    """Plot area-between-curves (state occupancy as stacked area).

    Classic PSM visualization showing survival curve partitioning.
    """
    _setup_style()

    model = psm_result.model
    if strategy is None:
        strategy = model.strategy_names[0]

    r = psm_result.results[strategy]
    times = r['times']
    trace = r['trace']

    n_states = model.n_states
    fills, borders = _get_state_colors(n_states)

    fig, ax = plt.subplots(figsize=figsize)

    # Stacked area: reverse order so first state on top
    cumulative = np.zeros(len(times))
    for i in range(n_states - 1, -1, -1):
        state_prob = trace[:, i]
        ax.fill_between(
            times, cumulative, cumulative + state_prob,
            color=fills[i], alpha=alpha, label=model.states[i],
            edgecolor=borders[i], linewidth=0.5,
        )
        cumulative = cumulative + state_prob

    # Overlay survival curves as lines
    surv_colors = ['#D32F2F', '#1565C0', '#2E7D32', '#FF6F00']
    for j, endpoint in enumerate(model.survival_endpoints):
        surv = r['survival_curves'][endpoint]
        sc = surv_colors[j % len(surv_colors)]
        ax.plot(times, surv, color=sc, linewidth=2.5, linestyle='-',
                label=f'S({endpoint})', zorder=5)

    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)

    if title is None:
        title = f'Partitioned Survival — {model.strategy_labels[strategy]}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=10)

    fig.tight_layout()
    return fig


def plot_psm_trace(
    psm_result, figsize: tuple = (10, 6), title: Optional[str] = None,
):
    """Plot PSM state occupancy as line chart (all strategies, panel per state)."""
    _setup_style()

    model = psm_result.model
    n_strategies = model.n_strategies
    colors = _get_strategy_colors(n_strategies)
    line_styles = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(1, model.n_states, figsize=figsize, sharey=True)
    if model.n_states == 1:
        axes = [axes]

    for s_idx, state in enumerate(model.states):
        ax = axes[s_idx]
        for st_idx, strategy in enumerate(model.strategy_names):
            r = psm_result.results[strategy]
            times = r['times']
            state_prob = r['trace'][:, s_idx]
            ls = line_styles[st_idx % len(line_styles)]
            ax.plot(times, state_prob, color=colors[st_idx], linestyle=ls,
                    linewidth=2, label=model.strategy_labels[strategy])

        ax.set_title(state, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (years)', fontsize=10)
        if s_idx == 0:
            ax.set_ylabel('Proportion', fontsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)

    if title is None:
        title = 'State Occupancy by Strategy'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    fig.tight_layout()
    return fig


def plot_psm_comparison(
    psm_result, endpoint: str, figsize: tuple = (10, 6),
    title: Optional[str] = None,
):
    """Compare a single survival endpoint across strategies with shaded area."""
    _setup_style()

    model = psm_result.model
    colors = _get_strategy_colors(model.n_strategies)

    fig, ax = plt.subplots(figsize=figsize)

    all_times = None
    all_surv = {}

    for s_idx, strategy in enumerate(model.strategy_names):
        r = psm_result.results[strategy]
        times = r['times']
        surv = r['survival_curves'][endpoint]
        all_surv[strategy] = surv
        if all_times is None:
            all_times = times

        ax.plot(times, surv, color=colors[s_idx], linewidth=2.5,
                label=model.strategy_labels[strategy])

    if model.n_strategies == 2:
        s1, s2 = model.strategy_names[:2]
        surv1 = all_surv[s1]
        surv2 = all_surv[s2]
        ax.fill_between(
            all_times, surv1, surv2,
            alpha=0.15, color=colors[1], label='Difference',
        )

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel(f'S({endpoint})', fontsize=12)

    if title is None:
        title = f'{endpoint} Survival Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best')
    fig.tight_layout()
    return fig


# =============================================================================
# Microsimulation Plots
# =============================================================================

def plot_microsim_trace(
    result,
    strategy: Optional[str] = None,
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
    **kwargs,
):
    """Plot mean state occupancy trace from microsimulation.

    Parameters
    ----------
    result : MicroSimResult
        Microsimulation result object.
    strategy : str, optional
        Plot a single strategy. Default: all strategies.
    """
    _setup_style()
    model = result.model
    strategies = [strategy] if strategy else model.strategy_names
    n_strat = len(strategies)

    if n_strat == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_strat, figsize=(figsize[0], figsize[1]),
                                 sharey=True)

    state_fills, state_borders = _get_state_colors(model.n_states)
    cycles = np.arange(model.n_cycles + 1) * model.cycle_length

    for s_idx, strat in enumerate(strategies):
        ax = axes[s_idx]
        trace = result.results[strat]['trace']

        for j in range(model.n_states):
            ax.plot(cycles, trace[:, j], color=state_borders[j],
                    linewidth=2, label=model.states[j])

        ax.set_xlabel('Time (years)', fontsize=11)
        if s_idx == 0:
            ax.set_ylabel('Proportion in State', fontsize=11)
        ax.set_title(model.strategy_labels[strat], fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, cycles[-1])

    if title is None:
        title = 'Microsimulation — State Occupancy Trace'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def plot_microsim_survival(
    result,
    figsize: tuple = (10, 7),
    title: Optional[str] = None,
    **kwargs,
):
    """Plot empirical survival curves from microsimulation.

    Parameters
    ----------
    result : MicroSimResult
        Microsimulation result object.
    """
    _setup_style()
    model = result.model
    colors = _get_strategy_colors(model.n_strategies)

    fig, ax = plt.subplots(figsize=figsize)

    surv_df = result.survival_curve()
    for s_idx, strat in enumerate(model.strategy_names):
        df_s = surv_df[surv_df['Strategy'] == model.strategy_labels[strat]]
        ax.plot(df_s['Time'].values, df_s['Survival'].values,
                color=colors[s_idx], linewidth=2.5,
                label=model.strategy_labels[strat])

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Proportion Alive', fontsize=12)

    if title is None:
        title = 'Microsimulation — Survival Curves'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    fig.tight_layout()
    return fig


def plot_microsim_outcomes(
    result,
    outcome: str = "qalys",
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    bins: int = 50,
    **kwargs,
):
    """Plot histogram of per-patient outcomes.

    Parameters
    ----------
    result : MicroSimResult
        Microsimulation result object.
    outcome : str
        "qalys", "cost", or "lys".
    bins : int
        Number of histogram bins.
    """
    _setup_style()
    model = result.model
    colors = _get_strategy_colors(model.n_strategies)

    fig, ax = plt.subplots(figsize=figsize)

    key_map = {
        'qalys': 'total_qalys',
        'cost': 'total_cost',
        'lys': 'total_lys',
    }
    label_map = {
        'qalys': 'QALYs',
        'cost': 'Total Cost',
        'lys': 'Life Years',
    }
    data_key = key_map.get(outcome, 'total_qalys')
    data_label = label_map.get(outcome, outcome)

    for s_idx, strat in enumerate(model.strategy_names):
        data = result.results[strat][data_key]
        ax.hist(data, bins=bins, alpha=0.5, color=colors[s_idx],
                label=f"{model.strategy_labels[strat]} "
                      f"(mean={data.mean():.2f})",
                edgecolor='white', linewidth=0.5)
        ax.axvline(data.mean(), color=colors[s_idx], linewidth=2,
                   linestyle='--', alpha=0.8)

    ax.set_xlabel(data_label, fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)

    if title is None:
        title = f'Microsimulation — Distribution of {data_label}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# CEAnalysis plots (multi-strategy comparison)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ce_frontier(cea, figsize=(10, 8), title=None, show_labels=True,
                     wtp=None, annotate_icer=True):
    """
    Plot CE plane with efficiency frontier.

    Dominated strategies shown with × markers, frontier strategies connected
    by a line, optionally with WTP threshold line.

    Parameters
    ----------
    cea : CEAnalysis
    figsize : tuple
    title : str, optional
    show_labels : bool
        Whether to label each point with strategy name.
    wtp : float, optional
        If provided, draw WTP threshold line from the reference strategy.
    annotate_icer : bool
        Whether to annotate ICER values on frontier segments.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_strategy_colors(cea.n_strategies)
    frontier = cea.frontier()

    markers = {"Ref": "o", "ND": "o", "D": "X", "ED": "s"}
    labels_map = {"Ref": "Frontier (ref)", "ND": "Frontier",
                  "D": "Dominated", "ED": "Ext. dominated"}
    sizes = {"Ref": 150, "ND": 150, "D": 100, "ED": 100}
    zorders = {"Ref": 5, "ND": 5, "D": 3, "ED": 3}

    # Plot each strategy
    plotted_labels = set()
    for _, row in frontier.iterrows():
        st = row["Status"]
        lbl = labels_map[st] if labels_map[st] not in plotted_labels else ""
        if lbl:
            plotted_labels.add(labels_map[st])

        color_idx = cea.strategies.index(row["Strategy"])
        facecolor = colors[color_idx] if st in ("Ref", "ND") else "white"
        edgecolor = colors[color_idx]

        ax.scatter(row["QALYs"], row["Cost"], marker=markers[st],
                   s=sizes[st], facecolors=facecolor, edgecolors=edgecolor,
                   linewidths=2, zorder=zorders[st], label=lbl)

        if show_labels:
            offset = (8, 8) if st in ("Ref", "ND") else (8, -12)
            ax.annotate(row["Strategy"], (row["QALYs"], row["Cost"]),
                        xytext=offset, textcoords="offset points",
                        fontsize=9, fontweight="bold" if st in ("Ref", "ND") else "normal",
                        color=edgecolor)

    # Draw frontier line
    frontier_rows = frontier[frontier["Status"].isin(["Ref", "ND"])].sort_values("Cost")
    if len(frontier_rows) > 1:
        ax.plot(frontier_rows["QALYs"], frontier_rows["Cost"],
                color="#333333", linewidth=2, linestyle="-", alpha=0.7,
                zorder=2, label="Efficiency frontier")

        # Annotate ICERs on segments
        if annotate_icer:
            for i in range(1, len(frontier_rows)):
                r0 = frontier_rows.iloc[i - 1]
                r1 = frontier_rows.iloc[i]
                mid_q = (r0["QALYs"] + r1["QALYs"]) / 2
                mid_c = (r0["Cost"] + r1["Cost"]) / 2
                icer_val = r1["ICER"]
                if np.isfinite(icer_val):
                    ax.annotate(f"ICER={icer_val:,.0f}",
                                (mid_q, mid_c), fontsize=8, color="#555555",
                                ha="center", va="bottom",
                                bbox=dict(boxstyle="round,pad=0.2",
                                         facecolor="lightyellow", alpha=0.8))

    # WTP threshold line
    if wtp is not None:
        ref_row = frontier_rows.iloc[0]
        q_range = ax.get_xlim()
        q_vals = np.array([ref_row["QALYs"], q_range[1]])
        c_vals = ref_row["Cost"] + wtp * (q_vals - ref_row["QALYs"])
        ax.plot(q_vals, c_vals, '--', color='gray', alpha=0.6, linewidth=1.5,
                label=f'WTP = {wtp:,.0f}/QALY')

    ax.set_xlabel("QALYs", fontsize=12)
    ax.set_ylabel("Cost ($)", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    if title is None:
        title = "Cost-Effectiveness Plane — Efficiency Frontier"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    return fig


def plot_nmb_curve(cea, wtp_range=(0, 150000), n_wtp=301,
                   figsize=(10, 7), title=None):
    """
    Plot NMB curves for each strategy across WTP thresholds.

    The optimal strategy at each WTP is where its NMB line is highest.
    Vertical dotted lines mark WTP values where the optimal strategy changes.

    Parameters
    ----------
    cea : CEAnalysis
    wtp_range : tuple
    n_wtp : int
    figsize : tuple
    title : str, optional
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_strategy_colors(cea.n_strategies)

    nmb_data = cea.nmb_curve(wtp_range=wtp_range, n_wtp=n_wtp)
    wtp_vals = nmb_data["WTP"].values

    for j, strat in enumerate(cea.strategies):
        ax.plot(wtp_vals, nmb_data[strat].values,
                color=colors[j], linewidth=2, label=strat)

    # Find switch points (where optimal strategy changes)
    nmb_matrix = np.column_stack([nmb_data[s].values for s in cea.strategies])
    optimal_idx = np.argmax(nmb_matrix, axis=1)
    switches = np.where(np.diff(optimal_idx) != 0)[0]
    for sw in switches:
        wtp_sw = (wtp_vals[sw] + wtp_vals[sw + 1]) / 2
        ax.axvline(wtp_sw, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.annotate(f'WTP ≈ {wtp_sw:,.0f}', (wtp_sw, ax.get_ylim()[1]),
                    fontsize=8, color='gray', ha='center', va='top',
                    rotation=90, xytext=(5, -5), textcoords='offset points')

    ax.set_xlabel("Willingness-to-Pay ($/QALY)", fontsize=12)
    ax.set_ylabel("Net Monetary Benefit ($)", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    if title is None:
        title = "Net Monetary Benefit by WTP Threshold"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    fig.tight_layout()
    return fig


def plot_ceaf(cea, wtp_range=(0, 150000), n_wtp=301,
              show_ceac=True, figsize=(10, 7), title=None):
    """
    Plot Cost-Effectiveness Acceptability Frontier (CEAF).

    Optionally overlays individual strategy CEAC curves (light) with
    the CEAF (bold black).

    Parameters
    ----------
    cea : CEAnalysis
        Must have PSA data.
    wtp_range : tuple
    n_wtp : int
    show_ceac : bool
        Whether to show individual CEAC curves behind the CEAF.
    figsize : tuple
    title : str, optional
    """
    ceaf_data = cea.ceaf(wtp_range=wtp_range, n_wtp=n_wtp)
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_strategy_colors(cea.n_strategies)
    wtp_vals = ceaf_data["WTP"].values

    # Individual CEAC curves
    if show_ceac:
        for j, strat in enumerate(cea.strategies):
            col = f"CEAC_{strat}"
            ax.plot(wtp_vals, ceaf_data[col].values,
                    color=colors[j], linewidth=1, alpha=0.4, linestyle='--',
                    label=f'{strat} (CEAC)')

    # CEAF = bold envelope
    ax.plot(wtp_vals, ceaf_data["CEAF"].values,
            color='black', linewidth=2.5, label='CEAF')

    # Shade regions by optimal strategy
    optimal_strats = ceaf_data["Optimal_Strategy"].values
    unique_regions = []
    i = 0
    while i < len(optimal_strats):
        start = i
        cur_strat = optimal_strats[i]
        while i < len(optimal_strats) and optimal_strats[i] == cur_strat:
            i += 1
        unique_regions.append((start, i - 1, cur_strat))

    for start, end, strat in unique_regions:
        color_idx = cea.strategies.index(strat)
        ax.fill_between(
            wtp_vals[start:end + 1], 0, ceaf_data["CEAF"].values[start:end + 1],
            alpha=0.08, color=colors[color_idx],
        )

    ax.set_xlabel("Willingness-to-Pay ($/QALY)", fontsize=12)
    ax.set_ylabel("Probability Cost-Effective", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    if title is None:
        title = "Cost-Effectiveness Acceptability Frontier (CEAF)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    fig.tight_layout()
    return fig


def plot_evpi(cea, wtp_range=(0, 150000), n_wtp=301,
              figsize=(10, 7), title=None, population=None):
    """
    Plot Expected Value of Perfect Information (EVPI) curve.

    Parameters
    ----------
    cea : CEAnalysis
        Must have PSA data.
    wtp_range : tuple
    n_wtp : int
    figsize : tuple
    title : str, optional
    population : float, optional
        If provided, also plot population EVPI (EVPI × population)
        on a secondary y-axis.
    """
    evpi_data = cea.evpi(wtp_range=wtp_range, n_wtp=n_wtp)
    fig, ax1 = plt.subplots(figsize=figsize)
    wtp_vals = evpi_data["WTP"].values
    evpi_vals = evpi_data["EVPI"].values

    color1 = COLORS['strategies'][0]
    color2 = COLORS['strategies'][1]

    ax1.plot(wtp_vals, evpi_vals, color=color1, linewidth=2.5,
             label='EVPI (per patient)')
    ax1.fill_between(wtp_vals, 0, evpi_vals, alpha=0.1, color=color1)
    ax1.set_xlabel("Willingness-to-Pay ($/QALY)", fontsize=12)
    ax1.set_ylabel("EVPI per Patient ($)", fontsize=12, color=color1)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.tick_params(axis='y', labelcolor=color1)

    if population is not None:
        ax2 = ax1.twinx()
        pop_evpi = evpi_vals * population
        ax2.plot(wtp_vals, pop_evpi, color=color2, linewidth=1.5,
                 linestyle='--', label=f'Population EVPI (N={population:,.0f})')
        ax2.set_ylabel(f"Population EVPI ($)", fontsize=12,
                        color=color2)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax2.tick_params(axis='y', labelcolor=color2)
        # Combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='best', fontsize=10)
    else:
        ax1.legend(loc='best', fontsize=10)

    if title is None:
        title = "Expected Value of Perfect Information (EVPI)"
    ax1.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# =============================================================================
# Budget Impact Analysis (BIA) Plots
# =============================================================================

def plot_budget_impact(
    bia,
    figsize: Tuple = (10, 6),
    title: Optional[str] = None,
    show_cumulative: bool = True,
    fmt: str = "$",
):
    """
    Bar chart of annual budget impact with optional cumulative line.

    Parameters
    ----------
    bia : BudgetImpactAnalysis
    figsize : tuple
    title : str, optional
    show_cumulative : bool
        Show cumulative budget impact line on secondary axis.
    fmt : str
        Currency symbol for axis labels.
    """
    _setup_style()
    fig, ax1 = plt.subplots(figsize=figsize)

    years = bia.years
    impact = bia.impact
    x = np.arange(len(years))

    # Colour bars by sign
    bar_colors = [
        COLORS['negative'] if v > 0 else COLORS['positive']
        for v in impact
    ]

    bars = ax1.bar(x, impact, color=bar_colors, width=0.6,
                   edgecolor='white', linewidth=0.8, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, impact):
        sign = "+" if val >= 0 else ""
        y_pos = bar.get_height()
        va = 'bottom' if val >= 0 else 'top'
        ax1.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{sign}{fmt}{val:,.0f}",
            ha='center', va=va, fontsize=9, fontweight='bold',
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Year {y}" for y in years])
    ax1.set_ylabel(f"Annual Budget Impact ({fmt})", fontsize=12)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, p: f"{fmt}{v:,.0f}")
    )

    if show_cumulative:
        ax2 = ax1.twinx()
        ax2.plot(x, bia.cumulative_impact, color=COLORS['strategies'][3],
                 marker='D', linewidth=2, markersize=7, zorder=4,
                 label='Cumulative')
        ax2.set_ylabel(f"Cumulative Impact ({fmt})", fontsize=12,
                       color=COLORS['strategies'][3])
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, p: f"{fmt}{v:,.0f}")
        )
        ax2.tick_params(axis='y', labelcolor=COLORS['strategies'][3])
        ax2.legend(loc='upper left', fontsize=10)

    if title is None:
        total = impact.sum()
        sign = "+" if total >= 0 else ""
        title = (
            f"Budget Impact Analysis "
            f"(Total: {sign}{fmt}{total:,.0f})"
        )
    ax1.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_budget_comparison(
    bia,
    figsize: Tuple = (10, 6),
    title: Optional[str] = None,
    fmt: str = "$",
):
    """
    Grouped bar chart comparing total costs: current vs new per year.

    Parameters
    ----------
    bia : BudgetImpactAnalysis
    figsize : tuple
    title : str, optional
    fmt : str
        Currency symbol.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    years = bia.years
    x = np.arange(len(years))
    width = 0.35

    c1 = COLORS['strategies'][0]
    c2 = COLORS['strategies'][1]

    ax.bar(x - width / 2, bia.total_current, width, label='Current Scenario',
           color=c1, edgecolor='white', linewidth=0.8, zorder=3)
    ax.bar(x + width / 2, bia.total_new, width, label='New Scenario',
           color=c2, edgecolor='white', linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Year {y}" for y in years])
    ax.set_ylabel(f"Total Cost ({fmt})", fontsize=12)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, p: f"{fmt}{v:,.0f}")
    )
    ax.legend(fontsize=10)

    if title is None:
        title = "Budget Comparison: Current vs New Scenario"
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_market_share(
    bia,
    figsize: Tuple = (12, 5),
    title: Optional[str] = None,
):
    """
    Side-by-side stacked area charts for market share evolution.

    Parameters
    ----------
    bia : BudgetImpactAnalysis
    figsize : tuple
    title : str, optional
    """
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    colors = _get_strategy_colors(bia.n_strategies)
    labels = [bia.strategy_labels[s] for s in bia.strategies]
    years = bia.years

    # Current scenario
    ax1.stackplot(
        years,
        [bia.market_share_current_matrix[i, :] * 100
         for i in range(bia.n_strategies)],
        labels=labels, colors=colors, alpha=0.85,
    )
    ax1.set_title("Current Scenario", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Market Share (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower left', fontsize=8)

    # New scenario
    ax2.stackplot(
        years,
        [bia.market_share_new_matrix[i, :] * 100
         for i in range(bia.n_strategies)],
        labels=labels, colors=colors, alpha=0.85,
    )
    ax2.set_title("New Scenario", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Year")
    ax2.legend(loc='lower left', fontsize=8)

    if title is None:
        title = "Market Share Evolution"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def plot_bia_detail(
    bia,
    scenario: str = "both",
    figsize: Optional[Tuple] = None,
    title: Optional[str] = None,
    fmt: str = "$",
):
    """
    Stacked bar chart of cost breakdown by strategy per year.

    Parameters
    ----------
    bia : BudgetImpactAnalysis
    scenario : str
        ``"current"``, ``"new"``, or ``"both"`` (side by side).
    figsize : tuple, optional
    title : str, optional
    fmt : str
        Currency symbol.
    """
    _setup_style()
    colors = _get_strategy_colors(bia.n_strategies)
    labels = [bia.strategy_labels[s] for s in bia.strategies]
    years = bia.years
    n = len(years)

    if scenario == "both":
        if figsize is None:
            figsize = (12, 6)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        _stacked_bar(ax1, years, bia.costs_current_disc, colors, labels,
                     "Current Scenario", fmt)
        _stacked_bar(ax2, years, bia.costs_new_disc, colors, labels,
                     "New Scenario", fmt)
    else:
        if figsize is None:
            figsize = (8, 6)
        fig, ax = plt.subplots(figsize=figsize)
        if scenario == "current":
            _stacked_bar(ax, years, bia.costs_current_disc, colors, labels,
                         "Current Scenario", fmt)
        else:
            _stacked_bar(ax, years, bia.costs_new_disc, colors, labels,
                         "New Scenario", fmt)

    if title is None:
        title = "Cost Breakdown by Strategy"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def _stacked_bar(ax, years, costs_matrix, colors, labels, subtitle, fmt):
    """Helper: draw a stacked bar chart on *ax*."""
    x = np.arange(len(years))
    bottom = np.zeros(len(years))
    for i in range(costs_matrix.shape[0]):
        ax.bar(x, costs_matrix[i, :], bottom=bottom, color=colors[i],
               label=labels[i], edgecolor='white', linewidth=0.6, width=0.6)
        bottom += costs_matrix[i, :]

    ax.set_xticks(x)
    ax.set_xticklabels([f"Year {y}" for y in years])
    ax.set_ylabel(f"Total Cost ({fmt})")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, p: f"{fmt}{v:,.0f}")
    )
    ax.set_title(subtitle, fontsize=11)
    ax.legend(fontsize=9)


def plot_bia_tornado(
    bia,
    sensitivities: Dict,
    figsize: Tuple = (10, 6),
    title: Optional[str] = None,
    fmt: str = "$",
):
    """
    Tornado diagram for BIA one-way sensitivity.

    Parameters
    ----------
    bia : BudgetImpactAnalysis
    sensitivities : dict
        ``{param_label: (low_value, high_value)}``.
    figsize : tuple
    title : str, optional
    fmt : str
        Currency symbol.
    """
    _setup_style()

    tornado_df = bia.tornado(sensitivities)
    base_val = tornado_df["Impact (Base)"].iloc[0]

    fig, ax = plt.subplots(figsize=figsize)

    n_params = len(tornado_df)
    y_pos = np.arange(n_params)

    for i, row in tornado_df.iterrows():
        lo = row["Impact (Low)"]
        hi = row["Impact (High)"]
        left = min(lo, hi)
        right = max(lo, hi)

        # Bar split at base
        if left < base_val:
            ax.barh(
                n_params - 1 - i,
                min(base_val, right) - left,
                left=left,
                height=0.6,
                color=COLORS['strategies'][0],
                edgecolor='white',
                linewidth=0.6,
            )
        if right > base_val:
            ax.barh(
                n_params - 1 - i,
                right - max(base_val, left),
                left=max(base_val, left),
                height=0.6,
                color=COLORS['strategies'][1],
                edgecolor='white',
                linewidth=0.6,
            )

    ax.axvline(base_val, color='black', linewidth=1, linestyle='-', zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tornado_df["Parameter"].values[::-1])
    ax.set_xlabel(f"Total Budget Impact ({fmt})", fontsize=12)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, p: f"{fmt}{v:,.0f}")
    )
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['strategies'][0], label='Low value'),
        Patch(facecolor=COLORS['strategies'][1], label='High value'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    if title is None:
        title = f"BIA Sensitivity (Base: {fmt}{base_val:,.0f})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig