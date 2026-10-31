# Graph clustering: results

These results come from the full-data run of `notebooks/graph_clustering.ipynb` (run tag `full_fromall_cut2014-01-01_u5b15`).
Every number below is in a CSV under `docs/results/full_fromall_cut2014-01-01_u5b15/`, and figures are in
`docs/figures/full_fromall_cut2014-01-01_u5b15/`. The reasons behind each choice are in
[`graph_clustering_decisions.md`](graph_clustering_decisions.md), referenced as D1–D25.

## Headline findings

1. **The recommenders built on our weighted graphs beat "readers also bought" on the long tail by a wide margin.**
   On 59,392 held-out users, with same-series books excluded, the community recommender on the co-liked / Jaccard /
   disparity graph reaches recall@10 = 0.0386 against 0.0353 for item-kNN on raw co-reader counts (+9.3%, 95% CI
   +6.2% to +12.7%). Tail recall is **3.1× higher** (0.0272 vs 0.0089), and mid recall +73% (0.0379 vs 0.0219). It
   pays for this on bestsellers: head recall is 0.0441 vs 0.0720. Item-kNN puts 81% of its recommendations on head
   books and 5% on the tail; the community recommender's split is 38% / 29% / 33%. Catalogue coverage is 82% vs 47%.
2. **A hybrid keeps most of both.** (1 − λ)·item-kNN + λ·community with λ = 0.5 (chosen on validation users) reaches
   recall@10 = 0.0410: +16% overall, +124% on the tail and only −11% on the head. All three differences are
   significant.
3. **The community step itself does not add accuracy over the backbone it is computed on.** Recommending from the
   same backbone without the community restriction (backbone-kNN) gives 0.0385, a difference of +0.0001 with a 95% CI
   of −0.0003 to +0.0004. The reason is that the communities follow the backbone closely: at resolution 1, 91% of
   the backbone's edge weight lies inside communities, so restricting candidates to the user's communities removes
   almost nothing. At higher resolutions more edges are cut and validation recall drops slightly. **The long-tail
   gain comes from the graph construction (Jaccard weighting plus a local backbone), not from the clustering.**
   This matters for how the report frames graph clustering; see "What this means for the report".
4. **The communities are meaningful.** Modularity is 0.84, and NMI against Amazon genres is 0.29 vs 0.06 for a
   random partition with the same sizes. 12 of the 39 large labelled communities mix genres in ways that make sense
   for readers, for example post-apocalyptic thrillers with prepper non-fiction, or clean romance with Christian fiction.
5. **The choice of weighting matters more than the choice of sparsifier.** Jaccard gives the highest modularity and the
   best balance between head and tail. PPMI is the best for the tail with top-k (tail recall 0.040) but almost useless
   for the head (0.006), because it gives popular books rare neighbours. Raw counts reproduce the popularity bias of
   item-kNN.

## 1. Data after preprocessing

| | reviews | users | books |
|---|---|---|---|
| Raw pull | 4,408,904 | 301,553 | 349,741 |
| After the iterative k-core (users ≥ 5, books ≥ 15; 15 iterations) | 2,287,187 | 165,151 | 51,981 |
| Train (before 2014-01-01) | 1,738,308 | 155,442 | 47,937 |
| Test (Jan–Jul 2014) | 548,879 | 102,580 | 44,093 |

- The k-core log, showing what each step removed, is in `kcore_log.csv`. The first book filter removes 1.57M reviews
  and 285k books. After that the steps shrink quickly.
- 80% of training reviews count as liked (rating ≥ 4).
- Figure: `01_data_distributions.png` (reviews per year with the cut-off; CCDFs of reviews per user and per book).

## 2. Graphs

**Bipartite matrix:** 155,442 × 47,937 CSR, with 1.74M read entries and 1.38M liked entries (density 0.023%).

**Heavy users** (D10): the heaviest training user has 2,049 books. 34 users (0.02%) have more than 500 books, and together
they create 17.5% of all co-occurrence pairs. Each user is weighted 1/log(1+d), so a median user (6 books) has
weight 0.51 and the heaviest 0.13.

| Graph (≥ 2 shared users) | edges | isolated books | mean degree |
|---|---|---|---|
| co-reader | 6,951,673 | 711 | 290 |
| co-liked | 4,043,544 | 1,312 | 169 |

Each co-occurrence product takes 4–6 s in blocks (D12).

**Weighting** (`04_weight_distributions.png`): the correlation between an edge's weight and the popularity of its two
endpoints is +0.34 for raw counts, −0.79 for Jaccard and −0.73 for PPMI. Raw counts favour popular pairs, while
Jaccard and PPMI over-correct. `example_neighbours.csv` shows the nearest neighbours of the most-read book,
*The Hunger Games*:

- Raw and Jaccard give its sequels, then other bestsellers.
- PPMI gives *Veganist*, *Good Christian Bitches* and *The Big-Ass Book of Crafts*, books with about 15 readers each.
  This is the rare-pair bias of PMI (D18).
- *Catching Fire* appears twice, once per edition. This is exactly what the LSH canonical mapping is meant to fix.

## 3. Sparsification

**Disparity filter at a fixed α** (`disparity_alpha_sweep.csv`, `05_alpha_sweep.png`), co-liked graph, α = 0.05:

| weighting | edges kept | books kept | giant component |
|---|---|---|---|
| raw | 2.2% | 47.2% | 34.7% |
| Jaccard | 2.6% | 79.7% | 64.6% |
| PPMI | 0.03% | 3.2% | 2.0% |

The same α gives very different backbones depending on the weighting. So for comparisons, α was matched to the edge
budget of top-10 (D13).

**Matched budget** (`backbones.csv`), co-liked graph:

| weighting | edges | α | books kept: disparity / top-10 | giant component: disparity / top-10 |
|---|---|---|---|---|
| raw | 361k | 0.17 | 81% / 100% | 76% / 97% |
| Jaccard | 309k | 0.12 | 95% / 100% | 91% / 97% |
| PPMI | 287k | 0.20 | 89% / 100% | 86% / 97% |

Both methods keep 4.7–8.9% of the edges. Top-10 keeps every book connected by construction. The disparity filter
drops books whose edges are all about equally weak, and it keeps more of the total weight with raw counts
(27% vs 19.5%).

## 4. Clustering

### Louvain resolution sweep (main variant: co-liked / Jaccard / disparity; `louvain_resolution_sweep.csv`, `06_resolution_sweep.png`)

| γ | modularity (γ = 1) | communities ≥ 10 books | largest community | stability across seeds: NMI / ARI |
|---|---|---|---|---|
| 0.5 | 0.817 | 52 | 31.8% | 0.82 / 0.79 |
| **1** | **0.837** | 68 | 14.9% | 0.83 / 0.72 |
| 2 | 0.829 | 109 | 8.6% | 0.84 / 0.74 |
| 5 | 0.777 | 198 | 3.7% | 0.85 / 0.65 |
| 10 | 0.727 | 329 | 2.1% | 0.86 / 0.65 |
| 20 | 0.657 | 557 | 1.4% | 0.87 / 0.62 |

All values are means over 5 seeds. Modularity peaks at γ = 1. The partitions are fairly stable across seeds
(NMI ≈ 0.83–0.87).

**Sanity checks:**

- Our modularity is 0.837432 on the main partition. igraph and networkx give the same value to 8 decimals.
- networkx's own Louvain reaches 0.835, and its partition has NMI 0.81 with igraph's.

### Spectral clustering (ours) vs Louvain on the giant component (43,685 books; `spectral_vs_louvain.csv`, `06_spectral_eigenvalues.png`)

| method | k | modularity | NMI vs genre | NMI, random partition |
|---|---|---|---|---|
| Louvain (γ = 1) | 124 | 0.836 | 0.278 | 0.035 |
| spectral | 20 | 0.712 | 0.237 | 0.015 |
| spectral | 40 | 0.751 | 0.253 | 0.025 |
| spectral | 60 | 0.736 | 0.267 | 0.032 |

Spectral clustering ran at full size (eigsh for 60 eigenvectors took about 5 s). It finds similar genre structure
but lower modularity, which is expected because Louvain optimises modularity directly.

### All 12 graph variants (`louvain_variants.csv`, γ = 1)

| source / weighting | modularity: disparity / top-10 | NMI: disparity / top-10 |
|---|---|---|
| co-reader raw | 0.72 / 0.78 | 0.30 / 0.29 |
| co-reader Jaccard | **0.84** / 0.81 | 0.28 / 0.29 |
| co-reader PPMI | 0.56 / 0.67 | 0.22 / 0.31 |
| co-liked raw | 0.74 / 0.78 | 0.32 / 0.27 |
| co-liked Jaccard | **0.84** / 0.81 | 0.29 / 0.28 |
| co-liked PPMI | 0.60 / 0.69 | 0.22 / 0.33 |

NMI is computed on the 5–7k books that have a genre (D6). The random baselines are 0.01–0.07.

## 5. Communities and genres (main partition; `community_genres.csv`, `07_community_sizes.png`)

- There are 412 communities (most of them are small fragments outside the giant component). 67 have at least 10 books
  and hold 97.9% of all books. The largest has 6,800 books (15%).
- 39 communities have at least 15 genre-labelled books. **12 of them are mixed**, meaning the second genre covers at
  least 25% of the labelled books. Examples:
  - **Post-apocalyptic + preparedness:** Mystery/Thriller 46% + Sci-fi 38%. Top titles are the *299 Days* series
    together with *Prepper's Instruction Manual*, so readers of collapse fiction also read survival non-fiction.
  - **Clean romance + Christian fiction:** Romance 38% + Christian 31% (*Longing for Home: A Proper Romance*,
    *Seeking Persephone*).
  - **Paranormal teen romance:** Teen & YA 41% + Romance 29% (*The Warlock King*, *Her Guardian Angel*).
  - **M/M romance:** Romance 60% + Gay & Lesbian 32% (*Frat Boy and Toppy*, *Nowhere Ranch*).
  - **Cozy witch mysteries:** Sci-fi & Fantasy 35% + Romance 26% (*A Charming Crime: A Magical Cures Mystery*).
  - The largest community (6,800 books) mixes History, Sci-fi and literary/political books (*Nineteen Eighty-Four*,
    *Disgrace*, *China, Inc.*). It is a broad "literary and political non-fiction" readership. Genre entropy is 0.72,
    but no single second genre reaches 25%, so it is not flagged as mixed.
- Genre-pure communities exist too, e.g. Romance 73% (1,990 books) and Christian fiction 65% (2,650 books).

## 6. Recommender (`recommender_results.csv`, `final_*.csv`, `significance_recall.csv`)

**Setup** (D19–D25):

- 74,715 test users. 20% (14,873) are used for validation and 59,842 for the final evaluation. 59,392 of those still have relevant items after same-series books are removed.
- 261,105 relevant items: 77k head, 72k mid and 112k tail.
- Popularity groups: head = 3,484 books (≥ 81 readers), mid = 12,443 books (30–80 readers), tail = 32,010 books.

**Validation choices** (`validation_grid.csv`, `validation_lambda.csv`, `08_hybrid_lambda.png`):

| within-community ranking, γ = 1 | validation recall@10 |
|---|---|
| strength (centrality, as in the plan) | 0.0034 |
| popularity | 0.0180 |
| **personal** | **0.0430** |
| item-kNN baseline (for reference) | 0.0393 |

The chosen setting was γ = 1 with the personal ranking (γ = 5 and γ = 20 were within 0.001). For the hybrid,
λ = 0.5 was chosen; λ = 0.7 gave the same overall recall with more tail.

**Final results** (held-out users, with books from series the user already read removed):

| recommender | recall@10 | NDCG@10 | hit@10 | head | mid | tail | coverage |
|---|---|---|---|---|---|---|---|
| most popular | 0.0077 | 0.0052 | 0.019 | 0.0210 | 0 | 0 | 0.04% |
| item-kNN (co-reader, "also bought") | 0.0353 | 0.0247 | 0.082 | **0.0720** | 0.0219 | 0.0089 | 47% |
| community, co-liked / Jaccard / disparity | 0.0386 | 0.0280 | 0.090 | 0.0441 | **0.0379** | **0.0272** | 82% |
| backbone-kNN, same graph | 0.0385 | 0.0279 | 0.090 | 0.0438 | 0.0377 | 0.0274 | 84% |
| hybrid λ = 0.5, same graph | **0.0410** | **0.0289** | **0.095** | 0.0644 | 0.0336 | 0.0198 | 73% |
| community, co-liked / Jaccard / top-10 | 0.0400 | 0.0290 | 0.092 | 0.0385 | 0.0390 | 0.0337 | 88% |
| community, co-reader / PPMI / top-10 | 0.0249 | 0.0185 | 0.060 | 0.0057 | 0.0184 | **0.0399** | 81% |

The full tables cover all 12 variants × {community, hybrid, backbone-kNN}, for recall, NDCG and hit, with and
without series.

**Paired bootstrap, recall@10, no same-series** (95% CI):

| comparison | all | head | mid | tail |
|---|---|---|---|---|
| community − item-kNN | +0.0033 [+0.0022, +0.0045] | −0.0279 | +0.0160 | +0.0183 [+0.0171, +0.0197] |
| hybrid − item-kNN | +0.0057 [+0.0048, +0.0066] | −0.0076 | +0.0117 | +0.0110 |
| community − backbone-kNN | +0.0001 [−0.0003, +0.0004] (n.s.) | n.s. | n.s. | n.s. |
| item-kNN − most popular | +0.0276 | +0.0509 | +0.0219 | +0.0089 |

**Series effect:** removing books from series the user already read lowers recall by 8% for item-kNN
(0.0382 → 0.0353) and by 11% for the community recommender (0.0436 → 0.0386). Series continuations inflate graph
methods a little more than co-purchase overlap, so the "no same-series" numbers are the honest ones to report.
`08_recall_by_group.png` shows recall per popularity group for the five key models.

## 7. Runtime and memory (`runtimes.csv`; MacBook, 16 GB RAM)

| step | time |
|---|---|
| preprocessing (DuckDB + k-core) | 6 s on the first run, then read from the Parquet cache |
| bipartite matrix | 5 s |
| co-occurrence graphs | 4–6 s each |
| weighting + sparsification (all 12 variants) | 20 s |
| Louvain sweep (6 resolutions × 5 seeds) | 13 s; each further variant takes about 1 s |
| networkx Louvain (check) | 41 s |
| spectral embedding, k = 60 | 4 s, plus 0.5–2.6 s of k-means per k |
| validation grid + λ sweep | 2 min |
| recommender evaluation (2 settings × 60k users) | about 2.4 min per graph variant (3 recommenders); most of the 34-min total |

Peak process memory was 3.2 GB.

## 8. Methods: from the course vs outside it

| From the course | Outside the course (implemented by us unless noted) |
|---|---|
| Louvain community detection (igraph / networkx library) | Jaccard and PPMI edge weighting of the projected bipartite graph |
| Spectral clustering on the normalised Laplacian (ours: Laplacian, eigsh, row-normalised k-means) | Disparity-filter backbone (Serrano, Boguñá & Vespignani 2009) and edge-budget matching |
| Modularity (ours, checked against igraph and networkx) | 1/log(1+d) user down-weighting (IDF / Adamic–Adar idea) |
| Sparse matrices, B^T B projection | Top-k evaluation (Hit / Recall / NDCG), popularity groups, paired bootstrap |
| NMI / ARI for comparing partitions | Community-restricted personal ranking and the hybrid recommender |

## What this means for the report

- **Supported:** graphs built from co-readers and co-liked books, weighted with Jaccard and sparsified to a local
  backbone, recommend long-tail books much better than purchase overlap alone (3× tail recall, significant). They
  also cover far more of the catalogue.
- **Not supported:** that *the clustering step* is what produces this gain. On the same backbone, restricting
  candidates to communities neither helps nor hurts (n.s.), because 91% of the backbone's weight is already inside
  communities. The report should say so plainly.
- **Where clustering does contribute:**
  - **Interpretability:** readable reader communities and cross-genre findings, with NMI 0.29 against genres.
  - **Compression:** the communities can stand in for the full backbone at no accuracy cost (412 groups vs
    309k edges), which is useful for explanations ("because you read books from the *prepper* community").
  - **Cold start (suggested for the cold-start teammate):** a new book with no readers can be placed in a community
    through the review-text graph and recommended to that community's readers. Here communities are needed, because
    a new book has no co-reader edges at all.
- **Limitations:**
  - It's the 2014 data.
  - Genre labels exist for only 18% of books.
  - Same-series detection is approximate (14% of books).
  - Reviews are used as a proxy for purchases.
  - Validation is a split of users, not of time.
  - Absolute recall values are low, which is typical for top-10 out of 48k books.
