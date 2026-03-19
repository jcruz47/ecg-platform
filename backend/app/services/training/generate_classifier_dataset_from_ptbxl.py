import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import wfdb
from tqdm import tqdm

PTBXL_ROOT = Path("data/ptbxl")
CSV_PATH = PTBXL_ROOT / "ptbxl_database.csv"
SCP_PATH = PTBXL_ROOT / "scp_statements.csv"

OUT_ROOT = Path("data/image_training/classifier")
IMG_SIZE = (14, 8)
DPI = 150

MAX_PER_CLASS = {
    "train": 2000,
    "val": 300,
    "test": 300,
}


def parse_scp_codes(text):
    if isinstance(text, dict):
        return text
    return ast.literal_eval(text)


def load_agg_df():
    agg_df = pd.read_csv(SCP_PATH, index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]
    return agg_df


def aggregate_diagnostic_superclasses(scp_codes: dict, agg_df: pd.DataFrame):
    classes = []
    for key in scp_codes.keys():
        if key in agg_df.index:
            diagnostic_class = agg_df.loc[key, "diagnostic_class"]
            if pd.notna(diagnostic_class):
                classes.append(diagnostic_class)
    return list(set(classes))


def label_from_superclasses(superclasses: list[str]) -> str | None:
    s = set(superclasses)

    # Solo NORM -> normal
    if s == {"NORM"}:
        return "normal"

    # Si tiene cualquier otra superclase diagnóstica -> abnormal
    abnormal_classes = {"MI", "STTC", "CD", "HYP"}
    if len(s & abnormal_classes) > 0:
        return "abnormal"

    # si no se puede decidir bien, se omite
    return None


def split_from_fold(strat_fold: int) -> str | None:
    if strat_fold in [1, 2, 3, 4, 5, 6, 7, 8]:
        return "train"
    if strat_fold == 9:
        return "val"
    if strat_fold == 10:
        return "test"
    return None


def count_existing(split: str, label: str) -> int:
    folder = OUT_ROOT / split / label
    folder.mkdir(parents=True, exist_ok=True)
    return len(list(folder.glob("*.png")))


def make_ecg_image(record_path: str, out_file: Path):
    record = wfdb.rdrecord(str(PTBXL_ROOT / record_path))
    signal = record.p_signal
    lead_names = record.sig_name

    fig, axes = plt.subplots(6, 2, figsize=IMG_SIZE)
    axes = axes.flatten()

    for i in range(min(len(lead_names), 12)):
        ax = axes[i]
        ax.plot(signal[:, i], linewidth=0.8)
        ax.set_title(lead_names[i], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    for j in range(len(lead_names), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No existe {CSV_PATH}")
    if not SCP_PATH.exists():
        raise FileNotFoundError(f"No existe {SCP_PATH}")

    agg_df = load_agg_df()

    df = pd.read_csv(CSV_PATH)
    df["scp_codes_dict"] = df["scp_codes"].apply(parse_scp_codes)
    df["superclasses"] = df["scp_codes_dict"].apply(lambda x: aggregate_diagnostic_superclasses(x, agg_df))
    df["label"] = df["superclasses"].apply(label_from_superclasses)
    df["split"] = df["strat_fold"].apply(split_from_fold)

    # quita filas sin etiqueta usable
    df = df[df["label"].notna()].copy()

    counts = {
        "train": {
            "normal": count_existing("train", "normal"),
            "abnormal": count_existing("train", "abnormal"),
        },
        "val": {
            "normal": count_existing("val", "normal"),
            "abnormal": count_existing("val", "abnormal"),
        },
        "test": {
            "normal": count_existing("test", "normal"),
            "abnormal": count_existing("test", "abnormal"),
        },
    }

    rows = df.sample(frac=1, random_state=42).to_dict("records")

    for row in tqdm(rows, desc="Generando imágenes ECG"):
        split = row["split"]
        if split is None:
            continue

        label = row["label"]
        if counts[split][label] >= MAX_PER_CLASS[split]:
            continue

        record_path = row["filename_lr"]   # usa records100
        ecg_id = row["ecg_id"]

        out_file = OUT_ROOT / split / label / f"{ecg_id}.png"
        if out_file.exists():
            continue

        try:
            make_ecg_image(record_path, out_file)
            counts[split][label] += 1
        except Exception as e:
            print(f"[WARN] error con {ecg_id}: {e}")

    print("Conteos finales:")
    print(counts)


if __name__ == "__main__":
    main()