# Visualizing Employee Attrition at IBM

A machine learning case study built around **visual model diagnostics** — using matplotlib and seaborn to explore, validate, and communicate an employee attrition analysis at every stage of the workflow.

The premise of this project is that in an analysis with 35 features and no clean separation between classes, the visualizations are not decoration on top of the results. They *are* the reasoning. Cluster count, linkage method, and final model selection were each decided by reading a chart, and the charts below are presented in the order those decisions were made.

**Dataset:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — 1,470 employees, 35 columns. Synthetic data created by IBM data scientists.

**Stack:** matplotlib, seaborn, pandas plotting (`parallel_coordinates`), SciPy (dendrograms)

---

## 1. Exploration — Small Multiples

A programmatically generated 5×2 subplot grid renders every categorical feature at once, with the loop pairing each axis to a column via `zip(axes.flatten(), cat_cols)`. Categories are sorted by frequency so the shape of each distribution reads consistently across panels.

![Categorical feature distributions](images/01_categorical_distributions.png)

This one figure replaces nine separate calls and surfaces the two features that would later be dropped — `Over18` and `StandardHours` were visibly single-valued. It also exposes the class imbalance in `Attrition` (16.1% Yes) that drives the modeling decision in section 8.

**Technique:** subplot grids, programmatic axis assignment, frequency-ordered categoricals.

---

## 2. Diagnostics — Correlation Heatmap

Before clustering, an annotated correlation matrix on the standardized features checks for redundancy.

![Correlation matrix heatmap](images/02_correlation_heatmap.png)

A diverging colormap (`coolwarm`) centered at zero is the deliberate choice here — correlation is a signed quantity, so a sequential palette would make −0.8 and 0.0 look equally unremarkable. Centering makes sign readable at a glance and direction distinguishable from magnitude. The tenure cluster (`YearsAtCompany` / `YearsWithCurrManager` / `YearsInCurrentRole`, 0.71–0.77) is immediately visible as a warm block, sitting just under the 0.8 threshold and documented as a watch item rather than acted on.

**Technique:** annotated heatmaps, diverging colormaps, `center=0` for signed data.

---

## 3. Structure — Truncated Dendrograms

Hierarchical clustering on 1,470 employees produces a dendrogram far too dense to read. A reusable `plot_dendrogram(df, method, color_threshold)` function handles this by truncating to the last *p* merged clusters, annotating each node with its distance, and printing member counts in parentheses on the leaves.

![Ward linkage dendrogram](images/03_dendrogram_ward.png)

The function was then run across **all seven linkage methods** with per-method color thresholds, since each operates on a different distance scale. Comparing the outputs is what eliminated four of them:

![Single linkage dendrogram](images/03b_dendrogram_single.png)

Single linkage (above) shows the textbook chaining failure — a long spine with individual employees peeling off, no meaningful group structure. Centroid and Median failed the same way. Ward and Complete were the only two producing balanced, cleanly nested branches.

**Technique:** parameterized plotting functions, truncation for high-*n* dendrograms, distance annotation, comparative small-multiple analysis across method variants.

---

## 4. Validation — Clustered Heatmap

With the field narrowed to Ward, a `clustermap` renders the actual employee-by-feature matrix with the dendrogram attached to the margin.

![Ward linkage clustermap](images/04_clustermap_ward.png)

This is the figure that settled the linkage choice. Silhouette scores were nearly tied and slightly favored Complete (0.1210 vs. Ward's 0.1153), but the clustermap shows Ward producing visibly cleaner horizontal banding across age, tenure, and income — evidence the score alone did not capture. Choosing the method on visual separation over a marginally better metric is the substantive judgment call in the project.

**Technique:** `sns.clustermap`, cluster-labeled axes, reading structure the summary statistic misses.

---

## 5. Interpretation — Centroid Heatmap and Parallel Coordinates

Two views of the same three centroids, chosen because each answers a different question.

![K-Means centroid heatmap](images/07_kmeans_centroid_heatmap.png)

The heatmap answers *"is this cluster high or low on this feature?"* — annotated z-scores on a diverging scale, scannable per cell.

![Parallel coordinates plot of centroids](images/08_parallel_coordinates.png)

The parallel coordinates plot answers *"where do the clusters actually separate?"* — following the three lines shows them running nearly flat and overlapping across `DailyRate`, `HourlyRate`, and `MonthlyRate`, then fanning sharply apart at `MonthlyIncome`, `TotalWorkingYears`, and the tenure variables. The features that define the segments are visually obvious.

This is what converted anonymous cluster IDs into the **Junior / Mid-Career / Senior** labels used throughout the rest of the analysis.

**Technique:** `pandas.plotting.parallel_coordinates`, paired complementary encodings of one dataset, standardized values to make disparate units comparable on a shared axis.

---

## 6. Parameter Selection — Elbow Plot

![Elbow plot for K-Means](images/06_elbow_plot.png)

Average within-cluster inertia across k=1–6. The curve bends at 3 and flattens after — independently confirming the cluster count that the Ward dendrogram had already suggested.

**Technique:** elbow method, cross-validating an unsupervised parameter against a second method.

---

## 7. Communication — Cluster Comparison Charts

Grouped bars compare Ward against K-Means cluster membership side by side:

![Cluster member counts by method](images/09_cluster_counts.png)

Near-identical bars across two independent algorithms is the cross-validation result, delivered in a form that needs no explanation.

Stacked bars then convert cluster membership into the business finding:

![Attrition percentage by cluster](images/10_attrition_pct_by_cluster.png)

![Attrition count by cluster](images/11_attrition_count_by_cluster.png)

Both rate *and* count are shown deliberately. Rate alone would suggest Junior and Senior are comparable concerns; count reveals Junior carries roughly 176 of the 237 departures. The pair together makes the case that percentage alone would understate.

**Technique:** grouped vs. stacked bar selection, manual bar offsetting for grouped comparison, `normalize=True` for proportional views, rate-and-volume pairing.

---

## 8. Model Evaluation — ROC Curves

Two logistic regression models were trained: a standard 60/40 split, and one on a training set oversampled to 50/50 to correct the class imbalance spotted back in section 1.

| | Standard Split | Oversampled |
|---|---|---|
| Accuracy | 85.9% | 74.8% |
| **Recall** | **36.3%** | **67.2%** |
| Precision | 67.3% | 35.2% |
| AUC | 0.84 | 0.81 |

![ROC curve — oversampled model](images/13_roc_oversampled.png)

The ROC curve is what justifies selecting the *less accurate* model. Both models have effectively equivalent AUC (0.84 vs 0.81), meaning their underlying ranking ability is comparable — the accuracy gap is a threshold artifact, not a quality difference. The steep initial rise confirms the oversampled model captures true positives at low false-positive cost. Given that a missed flight risk costs more than a false alarm, the model that nearly doubles recall wins, and the curve is the evidence.

**Technique:** ROC/AUC plotting, using curve shape to separate ranking quality from threshold effects.

---

## 9. Deployment — Risk Summary

![Active employee attrition risk](images/14_risk_pie.png)

![At-risk employees by cluster](images/15_at_risk_by_cluster.png)

Final output for a business audience: of 1,233 currently active employees, 285 (23%) exceed the 0.5 risk threshold — and the breakdown by cluster shows the same Junior concentration the clustering track found independently. Two analytical approaches converging on one conclusion, shown rather than argued.

---

## Visualization Skills Demonstrated

| Skill | Where |
|---|---|
| Subplot grids and small multiples | §1, §3 |
| Reusable parameterized plotting functions | §3 |
| Truncation strategies for high-cardinality data | §3 |
| Diverging colormaps and `center=0` for signed values | §2, §5 |
| Annotated heatmaps and clustermaps | §2, §4, §5 |
| Parallel coordinates for multivariate comparison | §5 |
| Grouped vs. stacked bar selection | §7 |
| Rate-and-volume pairing to avoid misleading conclusions | §7 |
| ROC/AUC curve interpretation | §8 |
| Charts as decision evidence, not illustration | throughout |

## Repository Contents

- `Attrition_at_IBM.ipynb` — main analysis notebook
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` — source dataset
- `images/` — exported figures

## Running the Analysis

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels dmba
jupyter notebook Attrition_at_IBM.ipynb
```

Run cells sequentially to reproduce every figure. Random seeds are fixed, so output is deterministic.

## References

- Hees, J. (2015, August 26). *SciPy hierarchical clustering and dendrogram tutorial.* Jörn's Blog. — dendrogram truncation approach used in §3
- Shmueli, G., Bruce, P. C., Yahav, I., Patel, N. R., & Lichtendahl, K. C. (2020). *Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python.* John Wiley & Sons.
- Dyerly, R. (2025, January 21). *The myth of replaceability: Preparing for the loss of key employees.* SHRM Executive Network.
