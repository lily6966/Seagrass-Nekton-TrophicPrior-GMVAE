from pathlib import Path
import json

import numpy as np
import pandas as pd

from process_nekton_seagrass import (
    DATA_DIR,
    SLED_CSV,
    TRAWL_CSV,
    SEED,
    build_feature_matrix,
    build_sample_table,
    choose_species_label,
    load_frame,
    make_splits,
)


FOCAL_LABELS = [
    "Penaeus aztecus",
    "Penaeus duorarum",
    "Callinectes sapidus",
    "Cynoscion nebulosus",
    "Lutjanus griseus",
    "Lutjanus synagris",
]

UNKNOWN_NICHE = {
    "Unknown species",
}

NICHE_ORDER = [
    "planktivore",
    "benthic_invertivore",
    "piscivore",
    "omnivore",
    "herbivore_detritivore",
    "shrimp_crab_invertivore",
    "unknown",
]


def trophic_group(label: str) -> str:
    if label in UNKNOWN_NICHE:
        return "unknown"

    herbivore_detritivore = {
        "Aplysia spp",
        "Argopecten irradians",
        "Archosargus probatocephalus",
        "Bursatella leachii",
        "Class Holothuroidea",
        "Crassostrea virginica",
        "Lagodon rhomboides",
        "Lytechinus variegatus",
        "Mugil cephalus",
        "Nicholsina usta",
        "Sparisoma spp",
    }
    if label in herbivore_detritivore:
        return "herbivore_detritivore"

    planktivore = {
        "Anchoa hepsetus",
        "Anchoa mitchilli",
        "Anchoa sp",
        "Brevoortia patronus",
        "Chloroscombrus chrysurus",
        "Harengula jaguana",
        "Menidia beryllina",
        "Menidia menidia",
        "Sardinella aurita",
        "Stellifer lanceolatus",
    }
    if label in planktivore:
        return "planktivore"

    shrimp_crab_invertivore = {
        "Achirus lineatus",
        "Ariopsis felis",
        "Bagre marinus",
        "Bairdiella chrysoura",
        "Calamus arctifrons",
        "Chaetodipterus faber",
        "Diplodus holbrookii",
        "Eucinostomus argenteus",
        "Eucinostomus gula",
        "Eucinostomus sp",
        "Gerreidae",
        "Haemulidae",
        "Haemulon plumieri",
        "Haemulon sp",
        "Leiostomus xanthurus",
        "Micropogonias undulatus",
        "Menticirrhus americanus",
        "Menticirrhus sp",
        "Orthopristis chrysoptera",
        "Pogonias cromis",
        "Sciaenidae",
        "Trinectes maculatus",
    }
    if label in shrimp_crab_invertivore:
        return "shrimp_crab_invertivore"

    piscivore = {
        "Acanthostracion quadricornis",
        "Ariopsis felis",
        "Bagre marinus",
        "Centropristis ocyurus",
        "Centropristis philadelphica",
        "Centropristis sp",
        "Centropristis striata",
        "Cynoscion arenarius",
        "Cynoscion sp",
        "Echeneis neucratoides",
        "Elops saurus",
        "Gymnura lessai",
        "Hypanus sabinus",
        "Lachnolaimus maximus",
        "Lagocephalus laevigatus",
        "Lutjanus analis",
        "Mycteroperca microlepis",
        "Ogcocephalus cubifrons",
        "Oligoplites saurus",
        "Ophidion holbrookii",
        "Paralichthyidae",
        "Paralichthys albigutta",
        "Paralichthys dentatus",
        "Paralichthys lethostigma",
        "Pomatomus saltatrix",
        "Prionotus scitulus",
        "Prionotus sp",
        "Sciaenops ocellatus",
        "Seriola rivoliana",
        "Serranus sp",
        "Serranus subligarius",
        "Sphoeroides nephelus",
        "Sphoeroides parvus",
        "Sphoeroides spengleri",
        "Sphyraena guachancho",
        "Sphyraena spp",
        "Synodus foetens",
        "Trichiurus lepturus",
    }
    if label in piscivore:
        return "piscivore"

    benthic_invertivore = {
        "Aluterus heudelotii",
        "Aluterus schoepfii",
        "Anarchopterus criniger",
        "Bascanichthys bascanium",
        "Bathygobius soporator",
        "Bathygobius sp",
        "Blenniidae",
        "Bothidae",
        "Callinectes ornatus",
        "Callinectes similis",
        "Caridea",
        "Chasmodes longimaxilla",
        "Chasmodes saburrae",
        "Chasmodes sp",
        "Chilomycterus schoepfi",
        "Citharichthys spilopterus",
        "Cosmocampus hildebrandi",
        "Ctenogobius boleosoma",
        "Ctenogobius sp",
        "Ctenogobius stigmaturus",
        "Cuapetes americanus",
        "Cuthona perca",
        "Diapterus auratus",
        "Diplectrum bivittatum",
        "Diplectrum formosum",
        "Epialtidae",
        "Etropus crossotus",
        "Evothodus lyricus",
        "Farfantepenaeus sp",
        "Favorinidae",
        "Fundulus grandis",
        "Fundulus heteroclitus",
        "Gambusia rhizophorae",
        "Gobiidae",
        "Gobiosoma bosc",
        "Gobiosoma robustum",
        "Gobiosoma sp",
        "Halicampus crinitus",
        "Halichoeres bivittatus",
        "Halichoeres sp",
        "Hippidae",
        "Hippocampus erectus",
        "Hippocampus zosterae",
        "Hippolytidae",
        "Histrio histrio",
        "Hypsoblennius hentz",
        "Hypsoblennius ionthas",
        "Latreutes fucorum",
        "Latreutes parvulus",
        "Latreutes sp",
        "Leander sp",
        "Libinia dubia",
        "Libinia emarginata",
        "Lolliguncula brevis",
        "Lucania parva",
        "Menippe adina",
        "Menippe mercenaria",
        "Menippe sp",
        "Metoporhaphis calcarata",
        "Microgobius gulosus",
        "Microgobius sp",
        "Microgobius thalassinus",
        "Microphis brachyurus",
        "Monacanthus ciliatus",
        "Monocanthus sp",
        "Monocanthus tuckeri",
        "Ogyrides alphaerostris",
        "Ogyrididae",
        "Opsanus beta",
        "Opsanus sp",
        "Palaemon floridanus",
        "Palaemon pugio",
        "Palaemon sp",
        "Palaemon vulgaris",
        "Palaemonidae",
        "Parablennius marmoreus",
        "Paraclinus fasciatus",
        "Paraclinus marmoratus",
        "Penaeidae",
        "Penaeus setiferus",
        "Periclimenes americanus",
        "Periclimenes longicaudatus",
        "Periclimenes sp",
        "Petrolisthes sp",
        "Pilumnus sayi",
        "Pitho sp",
        "Portunidae",
        "Portunus floridanus",
        "Portunus gibbesii",
        "Portunus sp",
        "Portunus vossi",
        "Processa sp",
        "Rimapenaeus sp",
        "Sicyonia laevigata",
        "Stephanolepis hispidus",
        "Stephanolepis sp",
        "Symphurus civitatium",
        "Symphurus plagiusa",
        "Symphurus sp",
        "Thor dobkini",
        "Thor sp",
        "Tozeuma carolinense",
        "Xanthidae",
    }
    if label in benthic_invertivore:
        return "benthic_invertivore"

    omnivore = {
        "Alpheidae",
        "Callinectes sapidus",
        "Cynoscion nebulosus",
        "Lutjanus griseus",
        "Lutjanus synagris",
        "Penaeus aztecus",
        "Penaeus duorarum",
        "Syngnathidae",
        "Syngnathus floridae",
        "Syngnathus louisianae",
        "Syngnathus pelagicus",
        "Syngnathus scovelli",
        "Syngnathus sp",
    }
    if label in omnivore:
        return "omnivore"

    return "unknown"


def build_group_features(raw_df: pd.DataFrame, sample_id_col: str) -> pd.DataFrame:
    df = raw_df.copy()
    df["sample_id"] = df[sample_id_col].astype(str)
    df["species_label"] = choose_species_label(df)
    df["species_abundance"] = pd.to_numeric(df["species_abundance"], errors="coerce").fillna(0.0)
    df = df[~df["species_label"].isin(FOCAL_LABELS)].copy()
    df["trophic_group"] = df["species_label"].map(trophic_group)
    df["is_present"] = (df["species_abundance"] > 0).astype(float)

    group_abund = (
        df.groupby(["sample_id", "trophic_group"])["species_abundance"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=NICHE_ORDER, fill_value=0.0)
    )
    group_abund.columns = [f"{c}__abundance_sum" for c in group_abund.columns]

    positive = df[df["species_abundance"] > 0].drop_duplicates(["sample_id", "species_label", "trophic_group"])
    group_rich = (
        positive.groupby(["sample_id", "trophic_group"])["species_label"]
        .nunique()
        .unstack(fill_value=0.0)
        .reindex(columns=NICHE_ORDER, fill_value=0.0)
    )
    group_rich.columns = [f"{c}__taxa_richness" for c in group_rich.columns]

    out = pd.concat([group_abund, group_rich], axis=1).reset_index()
    return out


def build_group_labels(raw_df: pd.DataFrame, sample_id_col: str) -> pd.DataFrame:
    df = raw_df.copy()
    df["sample_id"] = df[sample_id_col].astype(str)
    df["species_label"] = choose_species_label(df)
    df["species_abundance"] = pd.to_numeric(df["species_abundance"], errors="coerce").fillna(0.0)
    df = df[~df["species_label"].isin(FOCAL_LABELS)].copy()
    df["trophic_group"] = df["species_label"].map(trophic_group)
    df = df[df["species_abundance"] > 0].copy()
    df["present"] = 1.0

    grouped = (
        df.pivot_table(
            index="sample_id",
            columns="trophic_group",
            values="present",
            aggfunc="max",
            fill_value=0.0,
        )
        .reindex(columns=NICHE_ORDER, fill_value=0.0)
        .reset_index()
    )
    rename = {c: f"niche_label__{c}" for c in NICHE_ORDER}
    return grouped.rename(columns=rename)


def save_outputs(
    data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    label_cols: list[str],
    mapping_df: pd.DataFrame,
) -> None:
    suffix = "_fisheries6_plus_trophic_seagrass"
    stem = f"nekton_seagrass{suffix}"
    paths = {
        "data": DATA_DIR / f"{stem}_data.npy",
        "train": DATA_DIR / f"{stem}_train_idx.npy",
        "val": DATA_DIR / f"{stem}_val_idx.npy",
        "test": DATA_DIR / f"{stem}_test_idx.npy",
        "count": DATA_DIR / f"count{suffix}.txt",
        "features": DATA_DIR / f"feature_columns{suffix}.json",
        "labels": DATA_DIR / f"label_columns{suffix}.json",
        "info": DATA_DIR / f"processing_info{suffix}.json",
        "mapping": DATA_DIR / "trophic_mapping_nonfocal_species.csv",
    }

    np.save(paths["data"], data)
    np.save(paths["train"], train_idx)
    np.save(paths["val"], val_idx)
    np.save(paths["test"], test_idx)

    paths["count"].write_text(f"{len(feature_cols)}\n{len(label_cols)}\n")
    paths["features"].write_text(json.dumps(feature_cols, indent=2))
    paths["labels"].write_text(json.dumps(label_cols, indent=2))
    mapping_df.to_csv(paths["mapping"], index=False)
    paths["info"].write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_samples": int(data.shape[0]),
                "feature_dim": int(len(feature_cols)),
                "label_dim": int(len(label_cols)),
                "focal_labels": FOCAL_LABELS,
                "trophic_groups": NICHE_ORDER,
                "aggregation": "Non-focal taxa were aggregated into coarse trophic groups with summed abundance and taxa richness per haul/sample.",
                "mapping_note": "Trophic groups were inferred from taxon identity because no trophic lookup table was found in the repo.",
                "files": {k: str(v.name) for k, v in paths.items() if k in {"data", "train", "val", "test", "mapping"}},
            },
            indent=2,
        )
    )

    print("saved", paths["data"])
    print("mapping", paths["mapping"])
    print("samples", data.shape[0], "features", len(feature_cols), "labels", len(label_cols))
    print("splits", len(train_idx), len(val_idx), len(test_idx))
    print("nan in output", bool(np.isnan(data).any()))


def save_outputs_label_version(
    data: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_cols: list[str],
    label_cols: list[str],
    mapping_df: pd.DataFrame,
) -> None:
    suffix = "_fisheries6_plus_trophiclabels_seagrass"
    stem = f"nekton_seagrass{suffix}"
    paths = {
        "data": DATA_DIR / f"{stem}_data.npy",
        "train": DATA_DIR / f"{stem}_train_idx.npy",
        "val": DATA_DIR / f"{stem}_val_idx.npy",
        "test": DATA_DIR / f"{stem}_test_idx.npy",
        "count": DATA_DIR / f"count{suffix}.txt",
        "features": DATA_DIR / f"feature_columns{suffix}.json",
        "labels": DATA_DIR / f"label_columns{suffix}.json",
        "info": DATA_DIR / f"processing_info{suffix}.json",
        "mapping": DATA_DIR / "trophic_mapping_nonfocal_species.csv",
    }

    np.save(paths["data"], data)
    np.save(paths["train"], train_idx)
    np.save(paths["val"], val_idx)
    np.save(paths["test"], test_idx)

    paths["count"].write_text(f"{len(feature_cols)}\n{len(label_cols)}\n")
    paths["features"].write_text(json.dumps(feature_cols, indent=2))
    paths["labels"].write_text(json.dumps(label_cols, indent=2))
    mapping_df.to_csv(paths["mapping"], index=False)
    paths["info"].write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_samples": int(data.shape[0]),
                "feature_dim": int(len(feature_cols)),
                "label_dim": int(len(label_cols)),
                "focal_labels": FOCAL_LABELS,
                "niche_labels": [f"niche_label__{g}" for g in NICHE_ORDER],
                "aggregation": "Non-focal taxa were aggregated into trophic-niche presence labels per haul/sample. Predictors include only environmental, seagrass, and categorical site/time variables.",
                "mapping_note": "Trophic niches were inferred from taxon identity because no trophic lookup table was found in the repo.",
                "files": {k: str(v.name) for k, v in paths.items() if k in {"data", "train", "val", "test", "mapping"}},
            },
            indent=2,
        )
    )

    print("saved label-version", paths["data"])
    print("samples", data.shape[0], "features", len(feature_cols), "labels", len(label_cols))
    print("splits", len(train_idx), len(val_idx), len(test_idx))
    print("nan in output", bool(np.isnan(data).any()))


def main() -> None:
    sled = load_frame(SLED_CSV)
    trawl = load_frame(TRAWL_CSV)

    sled_samples = build_sample_table(sled, sample_id_col="pull_id", gear_is_sled=1)
    trawl_samples = build_sample_table(trawl, sample_id_col="tow_id", gear_is_sled=0)
    combined = pd.concat([sled_samples, trawl_samples], ignore_index=True, sort=False)

    label_cols = [c for c in sorted(FOCAL_LABELS) if c in combined.columns]
    combined[label_cols] = combined[label_cols].fillna(0.0)
    labels = combined[label_cols].astype(np.float64).to_numpy()

    trophic_sled = build_group_features(sled, "pull_id")
    trophic_trawl = build_group_features(trawl, "tow_id")
    trophic = pd.concat([trophic_sled, trophic_trawl], ignore_index=True, sort=False)

    combined = combined.merge(trophic, on="sample_id", how="left")
    trophic_cols = [c for c in combined.columns if c.split("__")[0] in NICHE_ORDER]
    combined[trophic_cols] = combined[trophic_cols].fillna(0.0)

    all_species_cols = [c for c in combined.columns if c not in label_cols and c in json.loads((DATA_DIR / "label_columns_seagrass.json").read_text())]
    features, feature_cols = build_feature_matrix(combined, ["sample_id"] + label_cols + all_species_cols)
    data = np.concatenate([labels, features], axis=1)
    train_idx, val_idx, test_idx = make_splits(len(combined))

    all_labels = json.loads((DATA_DIR / "label_columns_seagrass.json").read_text())
    mapping_rows = [
        {
            "species": label,
            "is_focal_label": int(label in FOCAL_LABELS),
            "trophic_group": "" if label in FOCAL_LABELS else trophic_group(label),
        }
        for label in all_labels
    ]
    mapping_df = pd.DataFrame(mapping_rows)

    save_outputs(data, train_idx, val_idx, test_idx, feature_cols, label_cols, mapping_df)

    niche_sled = build_group_labels(sled, "pull_id")
    niche_trawl = build_group_labels(trawl, "tow_id")
    niche_labels = pd.concat([niche_sled, niche_trawl], ignore_index=True, sort=False)

    label_version = combined.merge(niche_labels, on="sample_id", how="left")
    niche_label_cols = [f"niche_label__{g}" for g in NICHE_ORDER]
    label_version[niche_label_cols] = label_version[niche_label_cols].fillna(0.0)
    label_cols_full = label_cols + niche_label_cols
    labels_full = label_version[label_cols_full].astype(np.float64).to_numpy()
    env_feature_frame = label_version.drop(columns=trophic_cols, errors="ignore")
    features_label_version, feature_cols_label_version = build_feature_matrix(
        env_feature_frame,
        ["sample_id"] + label_cols_full + all_species_cols,
    )
    data_label_version = np.concatenate([labels_full, features_label_version], axis=1)
    save_outputs_label_version(
        data_label_version,
        train_idx,
        val_idx,
        test_idx,
        feature_cols_label_version,
        label_cols_full,
        mapping_df,
    )


if __name__ == "__main__":
    main()
