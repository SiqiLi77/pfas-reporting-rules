Supporting Information: figure source data, analysis outputs and scripts

Manuscript: Reporting Limits and Analyte Panels Shape Which Communities U.S. PFAS Drinking-Water Monitoring Can Identify

Contents (the layout mirrors the analysis project, so the scripts run where they are):
- 04_figures/source_data/ (41 CSV files): plotted values of Figures 1-4, S1-S16 and R1-R2, one file per panel or
  group of panels, named after the figure. For Figures S1-S13 and R1-R2 the column "source" gives the origin of each
  value: Tables S1-S16 and R1-R19, Texts S1-S10 and R1-R10, or the authors' code release (Text S11).
- 03_analysis/*.json (27 files) and *.log (15 files): analysis outputs. The numbers quoted in the manuscript,
  in Texts S11-S19 and in Tables S19-S29 are read from these files.
- 03_analysis/*.py and fetch_usdm.sh: analysis scripts. The table in Text S11 lists each script, its outputs and where
  they are used. Local modules imported by the scripts: pilot_common.py, si_digitize_lib.py.
- 03_analysis/author_release_v1_0_0/: the plotted values published with the authors' code release pfas-reporting-rules
  v1.0.0, used for Figures S1-S13 and R1-R2; SOURCE.json gives the commit and the SHA-256 of each file.
- requirements.txt: Python package versions used.
- MANIFEST.sha256: SHA-256 of every other file in this archive.

Raw data are not included. They are public: Text S11 lists each file with its source, size and SHA-256 (also in
03_analysis/input_checksums.json, with paths relative to 03_analysis/). Place them under 03_analysis/data/ as listed
there; fetch_usdm.sh downloads the Drought Monitor series.

Order of execution (each script reads the outputs of the scripts before it):
1. 03_analysis/load.py
2. 03_analysis/cache.py
3. 03_analysis/qa.py
4. 03_analysis/hash_inputs.py
5. 03_analysis/drought_prep.py
6. 03_analysis/pilot_e.py
7. 03_analysis/deep_lost.py
8. 03_analysis/deep_decide.py
9. 03_analysis/deep_family.py
10. 03_analysis/deep_auc.py
11. 03_analysis/deep_treat_season.py
12. 03_analysis/deep_drought.py
13. 03_analysis/deep_geo.py
14. 03_analysis/deep_ej.py
15. 03_analysis/deep_ej2.py
16. 03_analysis/deep_ej3.py
17. 03_analysis/deep_ej4.py
18. 03_analysis/deep_ej5_firth.py
19. 03_analysis/deep_misc.py
20. 03_analysis/calib_epa_table4.py
21. 03_analysis/deep_tables_extra.py
22. 03_analysis/deep_story.py
23. 03_analysis/fig_v2_02_data.py
24. 03_analysis/fig_r6_data.py
25. 03_analysis/si_redraw_data.py
26. 03_analysis/vintage_check.py
vintage_check.py additionally needs EPA's file of October 15, 2026 (Text S11).
