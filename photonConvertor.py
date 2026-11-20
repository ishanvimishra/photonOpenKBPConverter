
import json
import csv
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

DATA = Path("provided-data/train-pats")
OUTPUT = Path("converted-photon")
PRESCRIPTION = 70  

#convert openkbp names to ceci's format
NAMES = {
    "PTV70": "PTV_7000",
    "PTV63": "PTV_6300",
    "PTV56": "PTV_5600",
    "Brainstem": "Brainstem",
    "SpinalCord": "SpinalCord",
    "LeftParotid": "Parotid_L",
    "RightParotid": "Parotid_R",
    "Mandible": "Mandible",
    "Esophagus": "Esophagus",
    "Larynx": "Larynx"
}


def read(path):
    df = pd.read_csv(path, index_col=0)
    if "data" not in df or df.index.has_duplicates:
        raise ValueError(f"Invalid CSV: {path.name}")
    return df


def calculate(doses):
    values = {
        "Dmean": np.mean(doses),
        "D95": np.percentile(doses, 5),
        "D2": np.percentile(doses, 98),
        "Dmax": np.max(doses)
    }

    return [
        {
            "name": name,
            "value": float(value),
            "unit": "Gy",
            "pct_rx": float(value / PRESCRIPTION * 100)
        }
        for name, value in values.items()
    ]


def dvh(doses):
    steps = np.arange(
        0, np.ceil(max(doses) / 0.5) * 0.5 + 1, 0.5
    )

    return {
        "dose_unit": "Gy",
        "dose": steps.tolist(),
        "volume_pct": [
            float(np.mean(doses >= x) * 100)
            for x in steps
        ]
    }


def test():
    sample = np.array([10, 20, 30, 40, 50])
    result = {
        m["name"]: m["value"]
        for m in calculate(sample)
    }

    assert np.isclose(result["Dmean"], 30)
    assert np.isclose(result["D95"], 12)
    assert np.isclose(result["D2"], 49.2)
    assert np.isclose(result["Dmax"], 50)

    volumes = dvh(sample)["volume_pct"]
    assert [volumes[i] for i in [0, 20, 40, 60]] == [
        100, 100, 80, 60
    ]

    print("Synthetic test passed")

# converts patient's data into the json format
def convert(folder):
    read(folder / "ct.csv")
    dose = read(folder / "dose.csv")["data"]

    dimensions = pd.read_csv(
        folder / "voxel_dimensions.csv",
        header=None
    ).iloc[:, 0].to_numpy(dtype=float)

    if (len(dimensions) != 3 or
            not np.isfinite(dimensions).all() or
            np.any(dimensions <= 0) or
            dose.empty or
            not np.isfinite(dose).all() or
            (dose < 0).any()):
        raise ValueError("Invalid dose or voxel dimensions")

    structures = []
    metrics = []
    dvhs = []
    issues = []

    volume = np.prod(dimensions) / 1000

    for file in sorted(folder.glob("*.csv")):
        name = file.stem

        if name in {
            "ct", "dose", "voxel_dimensions",
            "possible_dose_mask"
        }:
            continue

        if name not in NAMES:
            issues.append(f"Unknown structure: {name}")
            continue

        voxels = read(file)

        if voxels.empty:
            issues.append(f"Empty structure: {name}")
            continue

        doses = dose.reindex(voxels.index).fillna(0).to_numpy()
        standard = NAMES[name]

        structures.append({
            "name": standard,
            "original_name": name,
            "type": "target" if name.startswith("PTV") else "oar",
            "volume_cc": float(len(voxels) * volume)
        })

        metrics.extend(
            {"structure": standard, **m}
            for m in calculate(doses)
        )

        dvhs.append({
            "structure": standard,
            **dvh(doses)
        })

    if not structures:
        raise ValueError("No supported structures found")

    if not any(s["type"] == "target" for s in structures):
        issues.append("Missing target")

    if not any(s["name"] == "PTV_7000" for s in structures):
        issues.append("Missing PTV70")

    patient = int(folder.name.split("_")[1])

    return {
        "schema_version": "0.1",
        "plan_id": f"PHO-OPENKBP-{patient:04d}",
        "modality": "photon",
        "source": "openkbp",
        "synthetic": False,
        "site": "head_and_neck",
        "prescription": {
            "dose": PRESCRIPTION,
            "unit": "Gy",
            "fractions": None
        },
        "rbe": 1.0,
        "structures": structures,
        "dvh": dvhs,
        "metrics": metrics,
        "prediction": None,
        "flags": [],
        "quality": {
            "status": "quarantined" if issues else "usable",
            "issues": issues
        },
        "provenance": {
            "created": date.today().isoformat(),
            "created_by": "photonConvertor.py"
        },
        "electron": None
    }


def save(name, rows):
    with open(OUTPUT / name, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    test()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        DATA.glob("pt_*"),
        key=lambda x: int(x.name.split("_")[1])
    )

    if not folders:
        raise ValueError("No patient folders found")

    log = []

    for folder in folders:
        patient = int(folder.name.split("_")[1])
        plan_id = f"PHO-OPENKBP-{patient:04d}"

        try:
            plan = convert(folder)

            with open(OUTPUT / f"{plan_id}.json", "w") as f:
                json.dump(plan, f, indent=2, allow_nan=False)

            status = plan["quality"]["status"]
            reason = "; ".join(plan["quality"]["issues"])

        except Exception as e:
            status = "failed"
            reason = str(e)

        log.append({
            "source_folder": folder.name,
            "plan_id": plan_id,
            "status": status,
            "reason": reason
        })

        print(folder.name, status)

    save("conversion_log.csv", log)

    counts = {
        status: sum(x["status"] == status for x in log)
        for status in ["usable", "quarantined", "failed"]
    }

    save("data_inventory.csv", [{
        "modality": "photon",
        "source": "openkbp",
        "cases_found": len(log),
        "cases_usable": counts["usable"],
        "quarantined": counts["quarantined"],
        "failed": counts["failed"],
        "blockers": "; ".join(sorted({
            x["reason"] for x in log if x["reason"]
        }))
    }])

    print(f"Processed {len(log)}: {counts}")


if __name__ == "__main__":
    main()


''' Photon vs. proton data differences: OpenKBP photon data provides CT, structure masks, and dose values in Gy.
Proton treatment data is expected to include similar information, potentially in different formats,
with dose represented in Gy(RBE). '''