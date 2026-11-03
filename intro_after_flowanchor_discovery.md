```latex
\section{Introduction}
\label{sec:intro}

Large pretrained video generators~\cite{wan2025,selfforcing2025}
now produce temporally coherent, high-fidelity footage.
Training-free video editing reuses these frozen generators to follow
textual or visual instructions without task-specific fine-tuning or
paired video-editing data~\cite{flowedit2025,tokenflow2024,streamedit2026}.
This setting is particularly appealing because paired editing videos
are expensive to collect, stronger base generators can be adopted
without retraining, and few-step generators enable efficient editing.

These methods steer a frozen generator between source preservation
and target adherence, typically through inversion, attention
manipulation, or velocity correction
~\cite{flowedit2025,tokenflow2024,streamedit2026,flowdirector}.
Many training-free editors control this trade-off through a single
global parameter shared across spatial locations and video frames.
Such global control overlooks the heterogeneous nature of editing
demand, which varies not only across editing tasks but also across
locations and frames within a video. A setting strong enough to
realize a substantial structural change, such as object insertion
or replacement, may also alter unrelated content; a more conservative
setting may preserve the scene but leave the requested transformation
under-realized.

Prior work addresses parts of this problem by controlling
\emph{where} editing should act. Training-free video methods use
user-provided masks or attention-derived spatial supports to localize
editing operations~\cite{objwiper,flowdirector}, while related
image-editing work infers editable regions from trajectory
divergence~\cite{fys}. These approaches use spatial support to
determine where editing is applied, often through binary masks
or softened variants that smooth transitions at region boundaries,
but do not express the degree of editability required at each
location. The complementary question of \emph{how much} to edit
has been explored by Differential Diffusion~\cite{diffdiff},
which enables continuous per-pixel edit strengths but requires
an externally specified change map.

More recently, FlowAnchor~\cite{flowanchor} addresses both spatial
localization and editing magnitude in inversion-free, flow-based
video editing. It refines cross-attention using spatial priors and
adaptively amplifies the source--target velocity difference according
to its instantaneous spatial contrast. While this provides
spatially adaptive editing within the flow trajectory, it does
not explicitly estimate local edit demand in a separate diagnostic
pass or translate such demand into the native controls of different
editing mechanisms.

We ask whether a training-free video editor can determine, before
the final edit, both \emph{where} to edit and \emph{how strongly},
and express this demand through its own native control mechanism.
To our knowledge, we are the first to use a separate, permissive
generation pass to automatically estimate a continuous,
location-dependent edit-demand field and allocate native editing
control accordingly.

Our key insight is to use a short, high-editability \emph{draft}
as a diagnostic probe of the editor's behavior. By temporarily
relaxing source preservation, we allow the requested transformation
to emerge. The resulting divergence between the draft and source
provides a graded signal of local edit demand, while source- and
target-branch attention supplies complementary spatial evidence
about the content involved in the requested edit. Together, these
signals produce a continuous, support-aware edit-demand field
that captures both where editing is needed and how much editability
each location requires.

We introduce \textbf{METHODNAME}, a plug-and-play method for
draft-guided adaptive edit strength in training-free video editing.
It operates in two passes: a short, permissive draft estimates
editable support and continuous per-token edit demand; a subsequent
pass uses this demand to adapt the host editor's native control
across spatial locations and frames. We normalize draft--source
divergence using a single fixed reference shared across videos
and host methods. We then convert the resulting demand into native
control values through a deterministic mapping derived from each
editor's update rule, rather than fitting a separate mapping to
rendered results. This allows the same demand-estimation procedure
to govern different editing mechanisms while leaving their
underlying operations unchanged.

We instantiate METHODNAME on two training-free video editors
covering complementary conditioning modalities and distinct
editing mechanisms. First, we consider visually prompted
StreamEdit~\cite{streamedit2026}, a few-step streaming editor
that uses an edited first frame to guide subsequent video frames.
Its source and target branches communicate through query--key
blending and source-feature injection. Second, we consider
text-conditioned Wan-Edit~\cite{five2025}, which follows the
FlowEdit formulation~\cite{flowedit} and performs inversion-free
editing through source- and target-conditioned velocity differences.
Its global control determines the denoising step at which the
editing correction begins. These two configurations allow us
to investigate whether the same draft-derived edit-demand
estimation transfers across different conditioning modalities
and native control mechanisms.

We evaluate METHODNAME on the official FiVE-Bench~\cite{five2025}
and FiVE-C, a minimally corrected annotation variant that retains
the original videos, masks, edit categories, and target concepts
while repairing documented textual inconsistencies. We compare
each adaptive editor against its own tuned global-control baseline
over a range of control settings at equal total sampling budget,
including the diagnostic draft. We further compare against
anchor-only and region-level alternatives to assess whether
draft-derived continuous edit demand provides benefits beyond
spatial localization alone.

Our contributions are:
\begin{itemize}
    \item \textbf{Diagnostic draft-based edit-demand estimation.}
    We introduce a separate, permissive generation pass that
    combines source- and target-branch attention with draft--source
    divergence to automatically estimate editable support and
    continuous, location-dependent edit demand, without a
    user-specified strength map.

    \item \textbf{Deterministic native-control allocation.}
    We introduce a shared absolute divergence normalization and
    a control conversion derived from each editor's update rule,
    translating the same edit-demand field into different native
    control mechanisms without fitting the mapping to generated
    results.

    \item \textbf{Cross-method evaluation.}
    We evaluate the proposed method on visually prompted StreamEdit
    and text-conditioned Wan-Edit, comparing each against its tuned
    global-control baseline and spatial-control alternatives at
    equal total sampling budget.
\end{itemize}
```