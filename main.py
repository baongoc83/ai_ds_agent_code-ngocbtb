import sys
import argparse
from pathlib import Path
from pipeline import AutoMLPipeline
from config import Config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Multi-Agent AutoML Pipeline: DataCleaner -> FeatureEngineer -> TrainModel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal - auto-detect everything (CSV, Parquet, Excel, etc.)
  python main.py data/train.csv TARGET
  python main.py data/train.parquet TARGET
  python main.py data/train.xlsx TARGET

  # Credit risk dataset (DPD30/90 default flag, MOB6/MOB12 window)
  python main.py data/train.csv TARGET \\
      --domain credit_risk --model-type binary_classification \\
      --entity-id SK_ID_CURR --keys SK_ID_CURR,MONTH_DT

  # Propensity model (buy / churn / interest)
  python main.py data/train.csv TARGET \\
      --domain propensity --model-type binary_classification

  # Fraud detection
  python main.py data/train.csv TARGET \\
      --domain fraud --model-type binary_classification

  # With column descriptions (HomeCredit format)
  python main.py data/train.csv TARGET \\
      --domain credit_risk \\
      --col-desc data/HomeCredit_columns_description.csv \\
      --col-name-field Row --col-desc-field Description --col-group-field Table

  # Split mode (train + oot)
  python main.py data/train.parquet TARGET --oot data/oot.parquet

  # Split mode — train + valid + oot all preserved exactly
  # Each agent FITS on train, TRANSFORMS valid + oot (no concat)
  python main.py data/train.parquet TARGET \\
      --valid data/valid.parquet --oot data/oot.parquet \\
      --domain credit_risk --entity-id customer_id --keys customer_id,snap_dt

  # Full options
  python main.py data/train.csv TARGET \\
      --domain credit_risk --model-type binary_classification \\
      --entity-id customer_id --keys customer_id,snapshot_date \\
      --col-desc data/columns.csv \\
      --col-name-field Row --col-desc-field Description --col-group-field Table \\
      --oot data/oot.csv
        """,
    )
    parser.add_argument(
        "input_path", nargs="?", default=None,
        help=(
            f"Path to input dataset (default: {Config.OUTPUT_DIR}/sample_data.csv). "
            "Supported formats: .csv, .tsv, .parquet, .orc, .feather, "
            ".xlsx/.xls/.xlsm, .json, and remote paths (s3://, gs://, az://)."
        ),
    )
    parser.add_argument(
        "target_column", nargs="?", default=None,
        help="Name of the binary target / label column (default: target)",
    )
    parser.add_argument(
        "--entity-id", metavar="COL", dest="entity_id_col", default=None,
        help="Primary entity identifier column, e.g. customer_id, SK_ID_CURR",
    )
    parser.add_argument(
        "--keys", metavar="COL1,COL2", dest="composite_key_cols", default=None,
        help="Comma-separated composite key columns. Agent 1 uses these for PK checks; "
             "Agent 3 extracts the date partner for OOT temporal split.",
    )
    parser.add_argument(
        "--valid", metavar="PATH", dest="valid_path", default=None,
        help="Pre-split VALID dataset. When provided (with or without --oot), "
             "switches to SPLIT MODE: Agent 1 FITS its cleaning pipeline on "
             "train only, then REPLAYS the same transforms on valid (and oot). "
             "Agent 2 follows the same fit-on-train / transform-on-valid pattern. "
             "Agent 3 concatenates the 3 cleaned + engineered files and reconstructs "
             "the exact user-defined train/valid/oot. Memory peak = size of the "
             "largest single partition (not the sum).",
    )
    parser.add_argument(
        "--oot", metavar="PATH", dest="oot_path", default=None,
        help="Pre-split OOT dataset (.csv / .parquet / .feather / .xlsx). "
             "Triggers SPLIT MODE (see --valid). Agent 1+2 transforms it "
             "using the spec fitted on train.",
    )
    parser.add_argument(
        "--sample-ratio", metavar="RATIO", type=float, dest="train_sample_ratio", default=None,
        help="Stratified sample fraction (0 < ratio < 1) applied to TRAIN only "
             "in split mode (--valid or --oot set). Valid / OOT kept intact. "
             "Useful for very wide datasets that OOM during cleaning. "
             "Example: --sample-ratio 0.3 keeps 30%% of train rows.",
    )
    parser.add_argument(
        "--no-prefilter", action="store_false", dest="prefilter", default=True,
        help="Disable column pre-filter (default: on). Pre-filter drops "
             "constant / near-constant / mostly-null columns from TRAIN "
             "before the LLM analysis; the same drops are captured in "
             "CleaningSpec and replayed on valid/oot during transform. "
             "Thresholds: PREFILTER_MAX_NULL_RATIO / PREFILTER_MAX_DOMINANT_RATIO in Config.",
    )
    parser.add_argument(
        "--no-drift-check", action="store_false", dest="check_distribution", default=True,
        help="Skip the Stage-0 PSI drift check between train / valid / oot (default: on). "
             "The check samples DRIFT_SAMPLE_N rows per partition and flags columns "
             "with PSI > DRIFT_PSI_THRESHOLD as drifting. Disable if you've already "
             "validated drift externally or want to re-run faster.",
    )
    parser.add_argument(
        "--no-auto-config", action="store_false", dest="auto_config", default=True,
        help="Disable auto-config from the raw input shape (default: on). When on, the "
             "pipeline profiles the input (rows/cols/numeric-categorical) and auto-derives "
             "feature-count caps and row-sampling sizes that are NOT already pinned via env. "
             "Explicitly-set env vars always win regardless of this flag.",
    )
    parser.add_argument(
        "--col-desc", metavar="PATH", dest="col_descriptions_path", default=None,
        help="Column descriptions file (.csv / .json / .parquet / .xlsx). "
             "Passed to Agent 2 to guide interaction feature creation.",
    )
    parser.add_argument(
        "--col-name-field", metavar="FIELD", dest="col_name_field", default="column_name",
        help="Field name for column names in col-desc file (default: column_name)",
    )
    parser.add_argument(
        "--col-desc-field", metavar="FIELD", dest="col_desc_field", default="description",
        help="Field name for descriptions in col-desc file (default: description)",
    )
    parser.add_argument(
        "--col-group-field", metavar="FIELD", dest="col_group_field", default=None,
        help="Field name for the source table / group in col-desc file (default: None)",
    )
    parser.add_argument(
        "--domain", metavar="DOMAIN", dest="domain", default="generic",
        choices=["credit_risk", "propensity", "fraud", "generic"],
        help="Feature engineering domain context for Agent 2 (default: generic). "
             "Options: credit_risk | propensity | fraud | generic",
    )
    parser.add_argument(
        "--model-type", metavar="TYPE", dest="model_type", default="binary_classification",
        choices=["binary_classification", "regression", "multiclass"],
        help="ML problem type (default: binary_classification). binary_classification: "
             "2 target values (0/1, yes/no, ... auto-encoded); multiclass: 3-50 labels "
             "(text or numbers, encoded 0..k-1, selection by macro AUC); regression: numeric target.",
    )
    parser.add_argument(
        "--product-type", metavar="PRODUCT", dest="product_type", default="generic",
        choices=["consumer_unsecured", "credit_card", "mortgage", "auto",
                 "overdraft", "bnpl", "sme", "generic"],
        help="Lending-product context for Agent 2's product-specific guidance "
             "(default: generic). Options: consumer_unsecured | credit_card | "
             "mortgage | auto | overdraft | bnpl | sme | generic",
    )
    parser.add_argument(
        "--workspace", dest="workspace", default="default",
        help="Workspace (per user / project): runs go to OUTPUT_DIR/workspaces/<name>/ "
             "('default' = OUTPUT_DIR itself). Clean up with tools/workspace_cleanup.py.",
    )
    parser.add_argument(
        "--calibration", dest="calibration", default=None, choices=["on", "off"],
        help="Probability calibration on CV out-of-fold predictions (default: CALIBRATION_ENABLED "
             "from .env). Ignored for regression.",
    )
    parser.add_argument(
        "--calibration-method", dest="calibration_method", default=None,
        choices=["isotonic", "sigmoid", "temperature", "ovr_isotonic"],
        help="binary: isotonic | sigmoid (default CALIBRATION_METHOD); "
             "multiclass: temperature | ovr_isotonic (default MULTICLASS_CALIBRATION_METHOD).",
    )
    return parser.parse_args()


def _print_metrics(final_metrics: dict) -> None:
    print(f"\nBest Model : {final_metrics.get('best_model', 'unknown')}")

    auc_keys = [
        ("cv_auc_mean",        "CV AUC (mean)   "),
        ("cv_auc_std",         "CV AUC (std)    "),
        ("valid_temporal_auc", "Valid temporal (IN-SAMPLE)"),
        ("valid_random_auc",   "Valid random (IN-SAMPLE)  "),
        ("valid_auc",          "Valid (IN-SAMPLE)         "),
        ("oot_auc",            "OOT             "),
        ("test_auc",           "Test (holdout)  "),
    ]
    if final_metrics.get("task") == "regression":
        print("\nModel Metrics (regression):")
        for key in ("cv_r2_mean", "cv_r2_std", "cv_rmse", "cv_mae",
                    "test_r2", "test_rmse", "test_mae", "oot_r2", "oot_rmse", "oot_mae"):
            val = final_metrics.get(key)
            if isinstance(val, float):
                print(f"  {key:<12}: {val:.4f}")
        auc_keys = []
    elif final_metrics.get("task") == "multiclass":
        print(f"\nModel Metrics (multiclass, {final_metrics.get('n_classes')} classes: "
              f"{final_metrics.get('classes')}):")
        for key in ("cv_auc_mean", "cv_accuracy", "cv_f1_macro", "cv_logloss",
                    "test_auc", "test_accuracy", "test_f1_macro", "test_logloss",
                    "oot_auc", "oot_accuracy", "oot_f1_macro", "oot_logloss"):
            val = final_metrics.get(key)
            if isinstance(val, float):
                print(f"  {key:<14}: {val:.4f}" + ("   (macro one-vs-rest)" if key.endswith(("auc", "auc_mean")) else ""))
        auc_keys = []
    else:
        print("\nModel Metrics:")
    for key, label in auc_keys:
        val = final_metrics.get(key)
        if isinstance(val, float):
            bar_len = int(max(0.0, min(1.0, val)) * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {label}: {val:.4f}  {bar}")

    best_params = final_metrics.get("best_params", {})
    if best_params:
        print("\nTop Hyperparameters:")
        for k, v in list(best_params.items())[:6]:
            print(f"  {k}: {v}")


def _print_benchmark(task: str) -> None:
    """Console version of the UI's 'Baseline & benchmark' + evaluation panel."""
    import json
    import target_encoding
    task = target_encoding.task_of(task or "classification")
    path = Path(Config.RUN_DIR) / f"{task}_benchmark.json"
    if not path.exists():
        return
    bm = json.loads(path.read_text(encoding="utf-8"))
    if task == "regression":
        from Agents.TrainModel import regression_benchmark as mod
    elif task == "multiclass":
        from Agents.TrainModel import multiclass_benchmark as mod
    else:
        from Agents.TrainModel import classification_benchmark as mod

    print("\n" + "-" * 65)
    print("  BASELINE & BENCHMARK  (all models fitted on train, same metrics)")
    print("-" * 65)
    for which, split in (("Validation", bm["eval_sets"].get("valid")), ("OOT/Test", bm["eval_sets"].get("holdout"))):
        if not split:
            continue
        table = mod.results_table(bm, split)
        if table.empty:
            continue
        print(f"\n{which} ({split}):")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if bm.get("failed"):
        print("\nNot run: " + "; ".join(f"{k} ({v})" for k, v in bm["failed"].items()))
    rec = bm.get("recommended_model") or {}
    if rec:
        print(f"\nRecommended model: {rec.get('note')}")
    icon = {"ok": "[OK]  ", "warn": "[WARN]", "bad": "[BAD] ", "info": "[INFO]"}
    print("\nEvaluation & recommendations:")
    for f in bm.get("findings", []):
        print(f"  {icon.get(f['severity'], '[--]  ')} {f['title']}")
        print(f"         {f['finding']}")
        print(f"         -> {f['action']}")
    print(f"\nFull report with charts + reading guide: {bm.get('report_md')}")


def _print_files(task: str = None) -> None:
    print(f"\nRun directory: {Config.RUN_DIR}")
    print("Generated files:")
    files = [
        # Reports + logs (always written)
        (Config.DATA_CLEANER_REPORT_PATH,        "Data cleaner report         — Agent 1"),
        (Config.FEATURE_ENGINEER_REPORT_PATH,    "Feature engineer report     — Agent 2"),
        (Config.MODEL_TRAINER_REPORT_PATH,       "Model trainer report        — Agent 3"),
        (Config.PSI_REPORT_PATH,                 "PSI drift report            — Agent 3"),
        (Config.STABILITY_REPORT_PATH,           "Feature stability report    — Agent 3"),
        (Config.SHAP_PSI_PRUNE_LOG_PATH,         "SHAP+PSI pruning log        — Agent 3"),
        (Config.SHAP_PLOT_PATH,                  "SHAP final bar plot         — Agent 3"),
        (Config.SHAP_BEESWARM_PATH,              "SHAP final beeswarm plot    — Agent 3"),
        (Config.SHAP_FEATURE_REPORT_PATH,        "SHAP top features CSV       — Agent 3"),
        (Config.SHAP_FINAL_MODEL_REPORT_PATH,    "SHAP combined markdown rpt  — Agent 3"),
        # Benchmark + model diagnostic charts (task-specific PNGs in charts/)
        *Config.chart_files(task),
        *[(f"{Config.RUN_DIR}/{t}_benchmark_report.md", f"{t.capitalize()} benchmark report — Agent 3")
          for t in ("regression", "classification", "multiclass")],
        (f"{Config.RUN_DIR}/target_mapping.json", "Target label encoding"),
        (Config.FINAL_REPORT_PATH,               "Full pipeline report"),
        (Config.EXECUTION_LOG_PATH,              "Agent execution log"),
        # Model deliverables
        (Config.FINAL_MODEL_PATH,                "Trained model artifact      — Agent 3"),
        (Config.FINAL_MODEL_CODE_PATH,           "Standalone inference code   — Agent 3"),
        # Per-agent replay scripts (re-apply each agent's transforms on new data)
        (Config.PIPELINE_PROCESS_DC_PATH,        "Replay script               — Agent 1"),
        (Config.PIPELINE_PROCESS_FE_PATH,        "Replay script               — Agent 2"),
        (Config.PIPELINE_PROCESS_FE_SPEC_PATH,   "Fitted FeatureSpec sidecar  — Agent 2"),
        (Config.PIPELINE_PROCESS_TM_PATH,        "Replay script               — Agent 3"),
    ]
    for filepath, description in files:
        if not Path(filepath).exists():
            continue
        print(f"  + {filepath:<70} {description}")
    if Config.KEEP_INTERMEDIATES:
        print(f"\n  [KEEP_INTERMEDIATES=true] clean_*.parquet / engineered_*.parquet kept in {Config.TMP_DIR}")


def main() -> None:
    print("=" * 65)
    print("  Multi-Agent AutoML Pipeline")
    print("  DataCleaner -> FeatureEngineer -> TrainModel")
    print("=" * 65)

    try:
        Config.validate()
        print("Configuration validated\n")
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    args = _parse_args()

    input_path = args.input_path or f"{Config.OUTPUT_DIR}/sample_data.csv"
    target_column = args.target_column or "target"
    composite_key_cols = (
        [c.strip() for c in args.composite_key_cols.split(",")]
        if args.composite_key_cols else None
    )
    col_desc_kwargs = (
        {
            "col_name_field": args.col_name_field,
            "col_desc_field": args.col_desc_field,
            "col_group_field": args.col_group_field,
        }
        if args.col_descriptions_path else None
    )

    print(f"Input        : {input_path}")
    print(f"Target       : {target_column}")
    if args.entity_id_col:
        print(f"Entity ID    : {args.entity_id_col}")
    if composite_key_cols:
        print(f"Composite key: {composite_key_cols}")
    if args.valid_path:
        print(f"Valid path   : {args.valid_path}  (pre-split mode)")
    if args.oot_path:
        print(f"OOT path     : {args.oot_path}")
    if args.train_sample_ratio is not None:
        print(f"Sample ratio : {args.train_sample_ratio}  (stratified, train only)")
    if not args.prefilter:
        print(f"Prefilter    : DISABLED")
    if not args.check_distribution:
        print(f"Drift check  : DISABLED")
    if not args.auto_config:
        print(f"Auto-config  : DISABLED")
    if args.col_descriptions_path:
        print(f"Col desc     : {args.col_descriptions_path}")
        if col_desc_kwargs:
            print(f"  name_field={args.col_name_field} | "
                  f"desc_field={args.col_desc_field} | "
                  f"group_field={args.col_group_field}")
    print(f"Domain       : {args.domain}")
    print(f"Product type : {args.product_type}")
    print(f"Model type   : {args.model_type}")
    if args.calibration_method:
        if args.calibration_method in ("temperature", "ovr_isotonic"):
            Config.MULTICLASS_CALIBRATION_METHOD = args.calibration_method
        else:
            Config.CALIBRATION_METHOD = args.calibration_method
    if args.calibration or args.calibration_method:
        print(f"Calibration  : {args.calibration or 'default'} ({args.calibration_method or 'default method'})")
    print()

    import workspace
    Config.OUTPUT_DIR = str(workspace.create(Config.OUTPUT_DIR, args.workspace))
    print(f"Workspace    : {args.workspace}  ({Config.OUTPUT_DIR})")
    pipeline = AutoMLPipeline()

    try:
        print("Starting the Multi-Agent Pipeline...\n")
        print("  Agent 1 — Data Cleaner   : audit quality, clean data")
        print("  Agent 2 — Feature Engineer: interaction features, encoding, selection")
        print("  Agent 3 — Train Model    : FLAML -> Optuna -> RFE -> PSI -> Stability -> SHAP+PSI -> Final")
        print()

        final_metrics = pipeline.run(
            input_path=input_path,
            target_column=target_column,
            col_descriptions_path=args.col_descriptions_path,
            col_descriptions_kwargs=col_desc_kwargs,
            entity_id_col=args.entity_id_col,
            composite_key_cols=composite_key_cols,
            valid_path=args.valid_path,
            oot_path=args.oot_path,
            train_sample_ratio=args.train_sample_ratio,
            prefilter=args.prefilter,
            check_distribution=args.check_distribution,
            domain=args.domain,
            model_type=args.model_type,
            product_type=args.product_type,
            auto_config=args.auto_config,
            calibration=None if args.calibration is None else args.calibration == "on",
        )

        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETE")
        print("=" * 65)

        _print_metrics(final_metrics)
        _print_benchmark(final_metrics.get("task"))
        _print_files(final_metrics.get("task"))

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
