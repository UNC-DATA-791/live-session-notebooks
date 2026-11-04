# casp_median_gdt.csv

Median of the best first-model GDT_TS per hard domain, for each CASP round.

- **Hard domains:** domain class is `FM`, `FM/TBM` or `FM-sp` in the round's domain summary
  (`https://predictioncenter.org/casp{N}/domains_summary.cgi`).
- **Score:** for each hard domain, take the highest GDT_TS among all groups' model 1
  (`..._1-Dn` rows). Then take the median across domains.
- **Scores from:** `https://predictioncenter.org/download_area/CASP{N}/` (`SUMMARY_TABLES/` for
  CASP11-12, `results/tables/` for CASP13-14).
- **Gaps:** CASP13 has 3 hard domains with no score table (`T0984-D0`, `T1000-D0`, `T1002-D0`).
  CASP10 has no domain classes on the site, so it is left out.
- **Check:** CASP14 gives 87.6. The AlphaFold2 median free-modelling GDT in the DeepMind
  CASP14 blog is 87.0.
