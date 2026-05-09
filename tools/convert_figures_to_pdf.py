from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"


FIGURES = [
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_sled" / "environment_plots_ecology_all_filtered" / "environment_predictors_heatmap_combined.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_trawl" / "environment_plots_ecology_all_filtered" / "environment_predictors_heatmap_combined.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_sled" / "seagrass_grouped_sum" / "seagrass_group_sum_importance_all_sites.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_trawl" / "seagrass_grouped_sum" / "seagrass_group_sum_importance_all_sites.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_sled" / "seagrass_grouped_sum" / "seagrass_feature_type_sum_importance_all_sites.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_trawl" / "seagrass_grouped_sum" / "seagrass_feature_type_sum_importance_all_sites.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_sled" / "seagrass_grouped_sum" / "seagrass_group_sum_importance_all_sites_shared_legend.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_trawl" / "seagrass_grouped_sum" / "seagrass_group_sum_importance_all_sites_shared_legend.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_sled" / "seagrass_grouped_sum" / "seagrass_feature_type_sum_importance_all_sites_shared_legend.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_trawl" / "seagrass_grouped_sum" / "seagrass_feature_type_sum_importance_all_sites_shared_legend.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_gear_split_environment_comparison_ecology_only.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_sled" / "environment_plots_ecology_top15" / "environment_predictors_heatmap_combined.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_trawl" / "environment_plots_ecology_top15" / "environment_predictors_heatmap_combined.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_sled" / "environment_plots_ecology_top15" / "environment_predictors_heatmap_all_sites.png",
    ROOT / "output" / "attribution" / "nekton_seagrass_fisheries6_plus_trophiclabels_site_trawl" / "environment_plots_ecology_top15" / "environment_predictors_heatmap_all_sites.png",
]


def convert_one(path: Path) -> Path:
    rel_parts = path.relative_to(ROOT / "output").parts
    pdf_name = "__".join(rel_parts).replace(".png", ".pdf")
    out_path = OUT / pdf_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.save(out_path, "PDF", resolution=300.0)
    return out_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fig in FIGURES:
        pdf = convert_one(fig)
        print(pdf)


if __name__ == "__main__":
    main()
