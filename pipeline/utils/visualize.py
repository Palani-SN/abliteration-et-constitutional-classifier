import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# =============================================================================
# ACTIVATION VISUALISATION
# Produces a single HTML report with three interactive Plotly views:
#   1. Heatmaps  — refuse_mean / accept_mean / raw_diff (layers x d_model)
#   2. Layer summary — norm(raw_diff) per layer
#   3. Per-layer detail — dropdown to inspect one layer at a time
# Called from compute_direction.py after the directions are computed.
# =============================================================================


def _to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


DIFF_THRESHOLD = 1.0  # dimensions with |raw_diff| <= this are masked to white


def plot_activation_analysis(
    refuse_mean,
    accept_mean,
    raw_diff,
    norms,
    out_path="activations/activation_analysis.html",
):
    rm = _to_np(refuse_mean)  # [L, D]
    am = _to_np(accept_mean)  # [L, D]
    rd = _to_np(raw_diff)     # [L, D]
    ns = _to_np(norms)        # [L]

    L, D = rm.shape
    layers = list(range(L))
    dims = list(range(D))

    # ── Figure 1: raw_diff heatmap (full width, weak dims masked) ────────────
    # Dimensions where |raw_diff| <= DIFF_THRESHOLD carry negligible signal;
    # masking them to NaN renders those cells white so strong-signal dims stand out.
    rd_masked = rd.copy()
    rd_masked[np.abs(rd_masked) <= DIFF_THRESHOLD] = 0.0
    abs_max = float(np.abs(rd_masked).max()) or 1.0

    fig1 = go.Figure(
        go.Heatmap(
            z=rd_masked,
            x=dims,
            y=layers,
            colorscale="RdBu",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title=dict(text="diff")),
            hovertemplate="layer=%{y}<br>dim=%{x}<br>raw_diff=%{z:.4f}<extra></extra>",
        )
    )
    fig1.update_layout(
        title_text=(
            f"raw_diff Heatmap — dims with |diff| > {DIFF_THRESHOLD} only "
            f"(rows: layers, cols: d_model dimensions)"
        ),
        xaxis_title="d_model dimension",
        yaxis_title="Layer",
        height=550,
    )

    # ── Figure 2: Layer summary — norm per layer ──────────────────────────────
    fig2 = go.Figure(
        go.Scatter(
            x=layers,
            y=ns.tolist(),
            mode="lines+markers",
            line=dict(color="royalblue", width=2),
            marker=dict(size=7),
            name="norm(raw_diff)",
            hovertemplate="layer %{x}<br>norm=%{y:.4f}<extra></extra>",
        )
    )
    fig2.update_layout(
        title_text="Refusal Direction Magnitude per Layer",
        xaxis_title="Layer index",
        yaxis_title="norm(refuse_mean - accept_mean)",
        height=380,
        hovermode="x unified",
    )

    # ── Figure 3: Per-layer detail with dropdown ──────────────────────────────
    fig3 = go.Figure()

    for layer_idx in range(L):
        visible = layer_idx == 0
        for data, label, color in [
            (rm, "refuse_mean", "red"),
            (am, "accept_mean", "green"),
            (rd, "raw_diff",    "royalblue"),
        ]:
            fig3.add_trace(
                go.Scatter(
                    x=dims,
                    y=data[layer_idx].tolist(),
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1),
                    visible=visible,
                    showlegend=(layer_idx == 0),
                    hovertemplate=f"dim=%{{x}}<br>{label}=%{{y:.4f}}<extra></extra>",
                )
            )

    # Each button makes exactly 3 traces visible (the chosen layer's trio)
    buttons = []
    for layer_idx in range(L):
        vis_flags = [False] * (L * 3)
        vis_flags[layer_idx * 3]     = True  # refuse_mean
        vis_flags[layer_idx * 3 + 1] = True  # accept_mean
        vis_flags[layer_idx * 3 + 2] = True  # raw_diff
        buttons.append(
            dict(
                label=f"Layer {layer_idx}",
                method="update",
                args=[
                    {"visible": vis_flags},
                    {"title": f"Per-layer detail - Layer {layer_idx}  (norm={ns[layer_idx]:.3f})"},
                ],
            )
        )

    fig3.update_layout(
        title_text=f"Per-layer detail - Layer 0  (norm={ns[0]:.3f})",
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=0.0,
                y=1.12,
                showactive=True,
            )
        ],
        xaxis_title="d_model dimension",
        yaxis_title="Activation value",
        height=500,
        hovermode="x unified",
    )

    # ── Combine all three figures into one HTML file ──────────────────────────
    divider = "\n<hr style='margin:40px 0;border:none;border-top:1px solid #ccc'>\n"
    html_body = divider.join(
        [
            pio.to_html(fig1, full_html=False, include_plotlyjs="cdn"),
            pio.to_html(fig2, full_html=False, include_plotlyjs=False),
            pio.to_html(fig3, full_html=False, include_plotlyjs=False),
        ]
    )

    html = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>"
        "<title>Activation Analysis</title>"
        "<style>body{font-family:sans-serif;padding:20px}</style>"
        "</head>\n<body>\n"
        "<h2>Activation Analysis - Refusal Direction</h2>\n"
        + html_body
        + "\n</body></html>"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved activation analysis -> {out_path}")
