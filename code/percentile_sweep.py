from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- USER SETTINGS ----------------

BASE_DIR = Path(r"D:\2nd_project_AUV_deepwater_fish_size_estimation\for_paper\end_to_end_measurement")
OUTPUT_DIR = BASE_DIR / "percentile_results"

PERCENTILES = np.arange(10, 100, 2)
DEVELOPMENT_IDS = {1, 2, 3}

ARRAY_KEYS = {
    "length": "standard_length_cm",
    "height": "height_cm",
}

FISH = pd.DataFrame({
    "fish_id": range(1, 10),
    "folder": [f"fish_{i}" for i in range(1, 10)],
    "reference_length_cm": [
        8.0, 8.4, 5.8, 3.8, 4.5, 4.0, 3.5, 4.0, 3.0
    ],
    "reference_height_cm": [
        3.1, 3.2, 4.5, 1.1, 1.5, 1.1, 1.0, 1.2, 0.9
    ],
})


# ---------------- DATA LOADING ----------------

def validate_settings():
    if FISH["fish_id"].duplicated().any():
        raise ValueError("Each fish_id must be unique.")

    if FISH["folder"].duplicated().any():
        raise ValueError("Each fish must have its own folder.")

    fish_ids = set(FISH["fish_id"])

    if not DEVELOPMENT_IDS.issubset(fish_ids):
        raise ValueError("Development IDs are missing from FISH.")

    if not DEVELOPMENT_IDS or DEVELOPMENT_IDS == fish_ids:
        raise ValueError(
            "Specify nonempty development and held-out groups."
        )

    references = FISH[
        ["reference_length_cm", "reference_height_cm"]
    ].to_numpy(dtype=float)

    if not np.isfinite(references).all() or (references <= 0).any():
        raise ValueError(
            "Reference dimensions must be finite and positive."
        )


def find_npz(folder):
    paths = sorted(folder.glob("*.npz"))

    if len(paths) != 1:
        raise ValueError(
            f"{folder}: expected exactly one NPZ file, "
            f"found {len(paths)}. Files: {[p.name for p in paths]}"
        )

    return paths[0]


def clean_measurements(raw, path, key):
    array = np.asarray(raw)

    if array.ndim != 1:
        raise ValueError(
            f"{path.name}, key {key!r}: expected a 1-D array, "
            f"received shape {array.shape}. "
            "Check the layout before flattening or pooling."
        )

    if array.dtype.kind not in "fiu":
        raise TypeError(
            f"{path.name}, key {key!r}: expected numeric data, "
            f"received {array.dtype}."
        )

    values = array.astype(float, copy=False)
    finite = np.isfinite(values)
    positive = values > 0
    valid = finite & positive

    counts = {
        "input_entries": int(values.size),
        "nonfinite_entries": int((~finite).sum()),
        "nonpositive_finite_entries": int(
            (finite & ~positive).sum()
        ),
        "valid_entries": int(valid.sum()),
    }

    if not valid.any():
        raise ValueError(
            f"{path}: no finite positive measurements in {key!r}."
        )

    return values[valid], counts


def load_all_fish():
    measurements = {}
    inventory_rows = []

    for fish in FISH.itertuples(index=False):
        folder = BASE_DIR / fish.folder

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        path = find_npz(folder)
        measurements[fish.fish_id] = {}

        with np.load(path, allow_pickle=False) as archive:
            missing = set(ARRAY_KEYS.values()) - set(archive.files)

            if missing:
                raise KeyError(
                    f"{path}: missing keys {sorted(missing)}. "
                    f"Available keys: {archive.files}"
                )

            for dimension, key in ARRAY_KEYS.items():
                values, counts = clean_measurements(
                    archive[key], path, key
                )

                measurements[fish.fish_id][dimension] = values

                inventory_rows.append({
                    "fish_id": fish.fish_id,
                    "folder": fish.folder,
                    "npz_file": path.name,
                    "dimension": dimension,
                    "array_key": key,
                    **counts,
                })

    return measurements, pd.DataFrame(inventory_rows)


# ---------------- PERCENTILE ANALYSIS ----------------

def make_percentile_estimates(measurements):
    rows = []

    for fish in FISH.itertuples(index=False):
        group = (
            "development"
            if fish.fish_id in DEVELOPMENT_IDS
            else "held_out"
        )

        for dimension in ARRAY_KEYS:
            values = measurements[fish.fish_id][dimension]
            reference = float(
                getattr(fish, f"reference_{dimension}_cm")
            )

            estimates = np.percentile(
                values,
                PERCENTILES,
                method="linear",
            )

            for percentile, estimate in zip(
                PERCENTILES, estimates
            ):
                error = float(estimate - reference)

                rows.append({
                    "fish_id": fish.fish_id,
                    "group": group,
                    "dimension": dimension,
                    "percentile": int(percentile),
                    "n_observations": int(values.size),
                    "reference_cm": reference,
                    "estimate_cm": float(estimate),
                    "signed_error_cm": error,
                    "absolute_error_cm": abs(error),
                    "absolute_percentage_error": (
                        100.0 * abs(error) / reference
                    ),
                })

    return pd.DataFrame(rows)


def summarize(frame):
    errors = frame["signed_error_cm"].to_numpy(dtype=float)

    return {
        "n_fish": int(frame["fish_id"].nunique()),
        "MAE_cm": float(np.mean(np.abs(errors))),
        "MAPE_percent": float(
            frame["absolute_percentage_error"].mean()
        ),
        "RMSE_cm": float(np.sqrt(np.mean(errors**2))),
        "bias_cm": float(np.mean(errors)),
    }


def select_percentiles(estimates):
    development = estimates.loc[
        estimates["group"].eq("development")
    ]

    rows = []

    for (dimension, percentile), frame in development.groupby(
        ["dimension", "percentile"], sort=True
    ):
        rows.append({
            "dimension": dimension,
            "percentile": int(percentile),
            **summarize(frame),
        })

    sweep = pd.DataFrame(rows)
    selected = {}

    for dimension in ARRAY_KEYS:
        candidates = sweep.loc[
            sweep["dimension"].eq(dimension)
        ].sort_values(["MAE_cm", "percentile"])

        selected[dimension] = int(
            candidates.iloc[0]["percentile"]
        )

    sweep["selected"] = [
        percentile == selected[dimension]
        for dimension, percentile in zip(
            sweep["dimension"], sweep["percentile"]
        )
    ]

    return sweep, selected


def evaluate_selected(estimates, selected):
    mask = np.zeros(len(estimates), dtype=bool)

    for dimension, percentile in selected.items():
        mask |= (
            estimates["dimension"].eq(dimension)
            & estimates["percentile"].eq(percentile)
        ).to_numpy()

    final = estimates.loc[mask].copy()
    metric_rows = []

    for dimension in ARRAY_KEYS:
        dimension_rows = final.loc[
            final["dimension"].eq(dimension)
        ]

        for group in ("development", "held_out", "all_fish"):
            frame = (
                dimension_rows
                if group == "all_fish"
                else dimension_rows.loc[
                    dimension_rows["group"].eq(group)
                ]
            )

            metric_rows.append({
                "dimension": dimension,
                "group": group,
                "selected_percentile": selected[dimension],
                **summarize(frame),
            })

    return final, pd.DataFrame(metric_rows)


def make_fish_table(final):
    columns = [
        "fish_id",
        "group",
        "reference_length_cm",
        "reference_height_cm",
    ]
    table = FISH.copy()
    table["group"] = np.where(
        table["fish_id"].isin(DEVELOPMENT_IDS),
        "development",
        "held_out",
    )
    table = table[columns]

    for dimension in ARRAY_KEYS:
        part = final.loc[
            final["dimension"].eq(dimension),
            [
                "fish_id",
                "percentile",
                "n_observations",
                "estimate_cm",
                "absolute_error_cm",
                "absolute_percentage_error",
            ],
        ].rename(columns={
            "percentile": f"{dimension}_percentile",
            "n_observations": f"{dimension}_n_observations",
            "estimate_cm": f"estimated_{dimension}_cm",
            "absolute_error_cm": f"{dimension}_absolute_error_cm",
            "absolute_percentage_error": f"{dimension}_APE_percent",
        })

        table = table.merge(
            part, on="fish_id", how="left", validate="one_to_one"
        )

    return table


# ---------------- FIGURES ----------------

def plot_development_sweep(sweep, selected):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    titles = {
        "length": "Standard length",
        "height": "Body height",
    }

    for ax, dimension in zip(axes, ARRAY_KEYS):
        frame = sweep.loc[
            sweep["dimension"].eq(dimension)
        ].sort_values("percentile")

        ax.plot(
            frame["percentile"],
            frame["MAE_cm"],
            marker="o",
        )
        ax.axvline(
            selected[dimension],
            color="tab:red",
            linestyle="--",
            label=f"Selected: P{selected[dimension]}",
        )
        ax.set(
            xlabel="Candidate percentile",
            ylabel="Development-set MAE (cm)",
            title=titles[dimension],
            xticks=np.linspace(10,100,10),
        )
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "development_percentile_sweep.png",
        dpi=300,
    )
    plt.close(fig)


# ---------------- MAIN ----------------

def run_analysis():
    validate_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    measurements, counts = load_all_fish()
    estimates = make_percentile_estimates(measurements)
    sweep, selected = select_percentiles(estimates)
    final, metrics = evaluate_selected(estimates, selected)
    fish_table = make_fish_table(final)

    outputs = {
        "observation_counts.csv": counts,
        "all_percentile_fish_estimates.csv": estimates,
        "development_percentile_sweep.csv": sweep,
        "selected_percentile_fish_results.csv": final,
        "fish_level_results.csv": fish_table,
        "group_metrics.csv": metrics,
    }

    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    settings = {
        "development_fish_ids": sorted(DEVELOPMENT_IDS),
        "held_out_fish_ids": sorted(
            set(FISH["fish_id"]) - DEVELOPMENT_IDS
        ),
        "candidate_percentiles": PERCENTILES.tolist(),
        "selection_metric": "development fish-level MAE in cm",
        "tie_rule": "lower percentile for an exact MAE tie",
        "percentile_method": "linear",
        "selected_percentiles": selected,
        "array_keys": ARRAY_KEYS,
        "pooling": (
            "Input arrays are treated as already pooled "
            "across both underwater cameras."
        ),
        "invalid_value_handling": (
            "Remove nonfinite and nonpositive entries; "
            "record counts; no other exclusions."
        ),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    (OUTPUT_DIR / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2),
        encoding="utf-8",
    )

    plot_development_sweep(sweep, selected)

    print("\nSelected development-set percentiles:")
    for dimension, percentile in selected.items():
        print(f"  {dimension}: P{percentile}")

    print("\nGroup metrics:")
    print(metrics.to_string(index=False, float_format="%.6f"))

    print("\nPer-fish results:")
    print(fish_table.to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")

    return {
        "measurements": measurements,
        "observation_counts": counts,
        "all_percentile_estimates": estimates,
        "development_sweep": sweep,
        "selected_percentiles": selected,
        "fish_results": final,
        "fish_table": fish_table,
        "group_metrics": metrics,
    }


if __name__ == "__main__":
    results = run_analysis()