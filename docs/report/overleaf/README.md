# Graph-clustering part for the Overleaf report

Files to upload into the Overleaf project (same folder as `main.tex`):

- `graph_clustering.tex`: our section, including the description-similarity graph (about 5,200 of the 11,000 characters allowed for the whole report).
- `discussion_limitations.tex`: the Discussion & Limitations section for the whole project (about 3,900 characters). If it goes in, delete the `Discussion` subsection at the end of `graph_clustering.tex`.
- `recall_by_group.pdf`: Figure "Recall@10 by book popularity group".

In `main.tex`:

1. In the preamble, make sure these packages are loaded: `\usepackage{amsmath,booktabs,graphicx}`
2. Where our part goes: `\input{graph_clustering}`; for the discussion, `\input{discussion_limitations}`
3. If our part sits under a larger section, change `\section` to `\subsection` (and `\subsection` to `\subsubsection`) in `graph_clustering.tex`.

The shared data description (the user-based pull, cut-off and k-core) belongs in the group's data section. The first paragraph of our section repeats only the two lines the graph part needs, so cut it if the data section already covers them.

Numbers come from `docs/results/full_fromall_cut2014-01-01_u5b15/`; see `docs/graph_clustering_results.md`.
