"""Interactive, numerically stable two-sample InfoNCE temperature explorer.

Run from the repository root:
    streamlit run demo/app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(page_title="InfoNCE Temperature Explorer", page_icon="🔬", layout="centered")

st.title("InfoNCE Temperature Explorer")
st.caption("Contrastive Representation Lab · MoCo / InfoNCE fundamentals")

st.markdown(
    r"""
For a **single positive** and a **single negative** with cosine similarities
$s^+$ and $s^-$, the InfoNCE loss is

$$
\mathcal{L} = -\log\frac{e^{s^+/\tau}}{e^{s^+/\tau}+e^{s^-/\tau}}
= \log\!\left(1+e^{(s^- - s^+)/\tau}\right).
$$

The temperature $\tau>0$ controls how strongly similarity differences affect the loss.
This is a **teaching example**, not the full multi-negative MoCo training objective.
"""
)

mode = st.radio("View", ["Single temperature", "Compare temperatures"], horizontal=True)
negative_sim = st.slider("Negative cosine similarity (s⁻)", -1.0, 1.0, 0.50, 0.05)
positive_sim = np.linspace(-1.0, 1.0, 401)

# logaddexp(0, x) implements log(1 + exp(x)) without numerical overflow.
def infonce_loss(pos_sim: np.ndarray | float, neg_sim: float, tau: float):
    return np.logaddexp(0.0, (neg_sim - pos_sim) / tau)

fig, ax = plt.subplots(figsize=(8, 4.7), dpi=120)

if mode == "Single temperature":
    tau = st.slider("Temperature (τ)", 0.05, 1.0, 0.20, 0.05)
    ax.plot(positive_sim, infonce_loss(positive_sim, negative_sim, tau),
            linewidth=2.5, label=f"τ = {tau:.2f}")
    sample_positive = st.slider("Inspect positive similarity (s⁺)", -1.0, 1.0, 0.80, 0.05)
    selected_loss = float(infonce_loss(sample_positive, negative_sim, tau))
    positive_probability = float(np.exp(-selected_loss))
    ax.scatter([sample_positive], [selected_loss], s=65, zorder=4)
    c1, c2 = st.columns(2)
    c1.metric("P(positive)", f"{positive_probability:.1%}")
    c2.metric("InfoNCE loss", f"{selected_loss:.4f}")
else:
    selected_temps = st.multiselect(
        "Temperatures to compare",
        options=[0.05, 0.07, 0.10, 0.20, 0.35, 0.50, 1.00],
        default=[0.07, 0.20, 0.50],
        format_func=lambda x: f"τ = {x:g}",
    )
    if not selected_temps:
        st.info("Choose at least one temperature to display its curve.")
    for tau in selected_temps:
        ax.plot(positive_sim, infonce_loss(positive_sim, negative_sim, tau),
                linewidth=2, label=f"τ = {tau:g}")

ax.axvline(negative_sim, linestyle="--", linewidth=1.3, alpha=0.7,
           label="s⁺ = s⁻ (loss = ln 2)")
ax.set_xlabel("Positive cosine similarity (s⁺)")
ax.set_ylabel("InfoNCE loss")
ax.set_xlim(-1.0, 1.0)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
ax.legend(loc="upper right")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.info(
    "Sanity check: when s⁺ = s⁻, the two candidates are equally likely, "
    "so the loss is ln(2) ≈ 0.6931 for every temperature."
)
st.markdown(
    "**Interpretation:** a smaller τ sharpens the contrast. If the positive "
    "is *less* similar than the negative, smaller τ can increase the loss "
    "dramatically. Smaller is **not always better** for training."
)


