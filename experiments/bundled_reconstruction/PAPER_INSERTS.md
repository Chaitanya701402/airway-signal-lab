# Figures and text for the research paper

Copy these three generated figures into the paper's figures folder before
compiling it:

| Figure | What it shows |
| --- | --- |
| `pipeline_overview.png` | Cleaning, source-group splitting, controlled synthesis and evaluation flow. |
| `ablation_comparison.png` | Fair E0--E3 comparison on the same held-out source groups. |
| `confusion_matrix_E2.png` | The E2 model's held-out classification errors. |

Suggested experimental-method text:

> We also conducted a reproducible small reconstruction experiment using the
> eight bundled breathing recordings. Payloads were decoded and cropped into
> fixed windows before any model fitting. Participant/source identifiers were
> split into train (105, 111, 125, 131), validation (147, 153), and held-out
> test (207, 216) groups. For each crop, one unmodified normal example and
> three controlled derived conditions were created: additive ambient noise,
> movable attenuation with an amplitude envelope, and fixed low-pass/gain
> attenuation. Log-Mel features were globally normalized using training
> features only. This reconstruction is a controlled engineering experiment;
> its generated conditions do not constitute clinical labels or clinical
> validation.

Suggested result text:

> On the held-out source groups, the CNN-only baseline (E0) obtained 76.39%
> accuracy and 75.60% macro F1. Adding a GRU (E1) reached 79.17% accuracy and
> 78.71% macro F1. The temporal-attention CNN (E2) was the strongest variant,
> with 98.61% accuracy and 98.61% macro F1, while the CNN--GRU--attention
> variant (E3) reached 88.19% accuracy and 87.59% macro F1. These comparisons
> are valid within the frozen reconstruction split only; they must not be
> compared numerically with the historical full-corpus results because the
> datasets and label construction differ.

LaTeX inserts (adjust filenames if the paper uses another figure directory):

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/pipeline_overview.png}
\caption{Cleaning, source-group split, controlled synthesis and evaluation in
the bundled reconstruction experiment.}
\label{fig:reconstruction-pipeline}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.86\linewidth]{figures/ablation_comparison.png}
\caption{Same-split held-out comparison of the four hybrid variants.}
\label{fig:ablation}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.72\linewidth]{figures/confusion_matrix_E2.png}
\caption{Held-out confusion matrix for E2, the strongest reconstruction
variant.}
\label{fig:e2-confusion}
\end{figure}
```
