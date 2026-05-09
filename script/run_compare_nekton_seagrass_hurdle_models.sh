OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HOME=$PWD/tmp/home XDG_CACHE_HOME=$PWD/tmp/cache MPLCONFIGDIR=$PWD/tmp/mpl python3 tools/plot_abundance_model_comparison.py \
  --env-metrics output/hurdle_abundance/nekton_seagrass_top20_env_seagrass/species_hurdle_metrics.csv \
  --coocc-metrics output/hurdle_abundance/nekton_seagrass_top20_coocc_abundance_seagrass/species_hurdle_metrics.csv \
  --output-dir output/hurdle_abundance/comparison_top20_env_vs_coocc
