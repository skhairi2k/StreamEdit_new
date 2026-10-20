# Absolute Divergence → Host-Aware Control Mapping

## Goal

Map a draft-derived, support-conditioned divergence value to a native control value without per-video normalization or manually tuned adaptive endpoints. Both hosts use the same fixed native range, \(c_{\min}=0\), \(c_{\max}=50\), but **the same divergence should not be mapped linearly to the same raw control value**: each host has a different control response.

## 1. Compute supported divergence

For each spatial token \(j\), let:

- \(D_j\geq 0\): divergence between source and high-editability draft features, computed with a fixed feature extractor and preprocessing;
- \(S_j\in[0,1]\): support from the union of source-phrase and target-phrase cross-attention maps, using the same frozen extraction rule for all hosts.

Compute

\[
E_j=S_jD_j.
\]

Use the same feature extractor, attention aggregation, and preprocessing for every video and host. Do not normalize each video by its own maximum or percentile.

## 2. Convert absolute divergence to normalized demand

Fix one global reference value \(E_{\mathrm{ref}}=0.55\) in the chosen divergence units. Then

\[
q_j=\operatorname{clip}\!\left(\frac{E_j}{E_{\mathrm{ref}}},0,1\right),
\qquad \operatorname{clip}(x,0,1)=\min(\max(x,0),1).
\]

Thus \(q_j\in[0,1]\) has an **absolute, globally fixed interpretation**: values below the reference retain their magnitude; values at or above the reference saturate at 1. A weak-divergence draft remains weak even if one token is the largest in that video. Freeze \(E_{\mathrm{ref}}\) before final evaluation; do not select it separately per host or per test video. The value 0.55 is the proposed reference and must be checked on development data using the exact divergence implementation.

## 3. Define each host's normalized control response

For host \(h\), define \(A_h(c,i)\in[0,1]\) as the native algorithm's **per-step edit-influence coefficient** at solver step \(i\) when its native control is \(c\). Derive it from the method's control equation or actual gating logic—not from the rendered output. For \(N\) solver steps, define the average exposure proxy

\[
Q_h(c)=\frac{1}{N}\sum_{i=1}^{N} A_h(c,i),\qquad 0\leq c\leq 50,
\]

and normalize it over the fixed endpoints:

\[
U_h(c)=\frac{Q_h(c)-Q_h(0)}{Q_h(50)-Q_h(0)}.
\]

Provided the denominator is positive and \(U_h\) is monotone, \(U_h(c)\in[0,1]\) describes the fraction of the host's available exposure range. It is a **control-mechanism proxy**, not a guarantee of equal perceptual edit strength across different models.

For each token, select the native control by inverse response:

\[
\boxed{c_{h,j}=\underset{c\in[0,50]}{\operatorname{argmin}}\;|U_h(c)-q_j|.}
\]

For continuous controls, evaluate \(U_h(c)\) on a fixed grid over \([0,50]\) and interpolate/invert it. For integer controls, enumerate the valid integers and select the closest response. Use one deterministic tie-breaking rule. Do not tune this mapping against final benchmark scores. If the measured response is non-monotone or the endpoint responses are identical, this inverse mapping is not justified and must be revised.

## 4. Host-specific response functions

### StreamEdit: \(c=\rho\)

StreamEdit's target-branch coefficient in the Q/K blend is

\[
A_{\mathrm{SE}}(\rho,i)=1-t_i^{\rho},
\]

where \(t_i\) is the actual normalized timestep used by the implementation. Compute

\[
Q_{\mathrm{SE}}(\rho)=\frac{1}{N}\sum_{i=1}^{N}(1-t_i^{\rho}),
\qquad
U_{\mathrm{SE}}(\rho)=\frac{Q_{\mathrm{SE}}(\rho)-Q_{\mathrm{SE}}(0)}{Q_{\mathrm{SE}}(50)-Q_{\mathrm{SE}}(0)}.
\]

For every token, set \(\rho_j\) to the value in \([0,50]\) whose \(U_{\mathrm{SE}}(\rho_j)\) is closest to \(q_j\). Use the actual discrete timestep list and the implementation's endpoint convention; do not substitute a continuous-time approximation for the final mapping.

### WanEdit: \(c=n_{\max}\)

Define \(A_{\mathrm{Wan}}(n_{\max},i)\) from WanEdit's **actual implementation** as the edit-influence indicator/coefficient at solver step \(i\) under that \(n_{\max}\). For a hard active/inactive edit gate, it is 1 when the edit branch is active and 0 otherwise. Then compute

\[
Q_{\mathrm{Wan}}(n_{\max})=\frac{1}{N}\sum_{i=1}^{N}A_{\mathrm{Wan}}(n_{\max},i),
\qquad
U_{\mathrm{Wan}}(n_{\max})=\frac{Q_{\mathrm{Wan}}(n_{\max})-Q_{\mathrm{Wan}}(0)}{Q_{\mathrm{Wan}}(50)-Q_{\mathrm{Wan}}(0)}.
\]

Enumerate valid integer \(n_{\max}\in\{0,1,\ldots,50\}\) and choose the value whose normalized exposure is closest to \(q_j\). Do **not** assume \(U_{\mathrm{Wan}}(c)=c/50\) unless the actual gate logic proves that relationship. Confirm the direction of the control (whether larger \(n_{\max}\) means more edit influence) from the code.

## 5. Interpretation and required checks

- \(q_j\) is the same fixed-scale, absolute edit-demand value for every host.
- \(c_{h,j}\) is host-specific because each host converts its control into edit influence differently.
- \(c_{\min}=0\) and \(c_{\max}=50\) define the candidate native range; they are not manually fitted per-token limits.
- The method assumes higher supported draft divergence should request greater edit exposure. Validate this assumption and the support maps on held-out development clips.
- Saturation at \(q=1\) loses distinctions among all divergences above \(E_{\mathrm{ref}}\); report the saturation fraction and freeze the reference before final tests.
- Keep the divergence-to-demand rule and host-response inversion fixed during evaluation. Compare adaptive and tuned-global baselines at equal total NFE, including the draft pass.

## References

- StreamEdit, *Training-Free Video Editing via Few-Step Streaming Video Generation*: https://arxiv.org/abs/2605.21466
- FiVE-Bench repository with Wan-Edit implementation: https://github.com/Harvard-AI-and-Robotics-Lab/FiVE-Bench
