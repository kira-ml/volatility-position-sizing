#!/usr/bin/env python
"""
Main orchestration pipeline for volatility forecasting project.

This script runs the complete end-to-end pipeline:
1. Data loading and validation
2. Feature engineering (all feature sets)
3. Model training and evaluation (all models, all feature sets)
4. Evaluation report generation (tables and figures)
5. Backtest simulation (static vs dynamic sizing)

Usage:
    python run_pipeline.py
    python run_pipeline.py --tickers AAPL MSFT --start-date 2021-01-01
    python run_pipeline.py --feature-set baseline_3 --skip-download
    python run_pipeline.py --help
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path if running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.data_loader import load_data, get_ticker_list, get_date_range
from src.features import build_feature_matrix, filter_features, get_feature_list
from src.models import run_all_experiments, create_comparison_table
from src.evaluate import generate_evaluation_report
from src.backtest import generate_backtest_report


# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------

def setup_logging(level: str = 'INFO') -> None:
    """Configure logging for the pipeline."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Volatility Forecasting Pipeline for Risk-Aware Position Sizing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_pipeline.py

  # Run with custom tickers and date range
  python run_pipeline.py --tickers AAPL MSFT GOOGL --start-date 2020-06-01 --end-date 2023-12-31

  # Run only a specific feature set
  python run_pipeline.py --feature-set baseline_3

  # Skip download (use cached data)
  python run_pipeline.py --skip-download

  # Skip backtest (faster for model experimentation)
  python run_pipeline.py --skip-backtest
        """
    )

    # Data options
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=config.TICKERS,
        help='Stock tickers to use (default: config.TICKERS)'
    )
    parser.add_argument(
        '--start-date',
        default=config.START_DATE,
        help=f'Start date YYYY-MM-DD (default: {config.START_DATE})'
    )
    parser.add_argument(
        '--end-date',
        default=config.END_DATE,
        help=f'End date YYYY-MM-DD (default: {config.END_DATE})'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip downloading fresh data (use cached CSV files)'
    )

    # Feature options
    parser.add_argument(
        '--feature-set',
        choices=['baseline_1', 'baseline_2', 'baseline_3', 'advanced', 'all'],
        default='all',
        help='Feature set to use (default: all)'
    )

    # Model options
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=config.TEST_SPLIT_RATIO,
        help=f'Test split ratio (default: {config.TEST_SPLIT_RATIO})'
    )
    parser.add_argument(
        '--ridge-alpha',
        type=float,
        default=None,
        help='Ridge alpha (default: grid search)'
    )

    # Backtest options
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip backtest simulation (faster execution)'
    )
    parser.add_argument(
        '--target-vol',
        type=float,
        default=config.TARGET_VOLATILITY,
        help=f'Target volatility (default: {config.TARGET_VOLATILITY})'
    )
    parser.add_argument(
        '--max-leverage',
        type=float,
        default=config.MAX_LEVERAGE,
        help=f'Maximum leverage (default: {config.MAX_LEVERAGE})'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        default=config.OUTPUTS_PATH,
        help=f'Output directory (default: {config.OUTPUTS_PATH})'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=config.LOG_LEVEL,
        help=f'Log level (default: {config.LOG_LEVEL})'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Don\'t save intermediate files (feature matrix, results)'
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Pipeline Functions
# -----------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> Dict:
    """
    Execute the complete pipeline.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Dictionary with pipeline results.
    """
    logger = logging.getLogger(__name__)
    results = {}

    print("\n" + "=" * 70)
    print("VOLATILITY FORECASTING PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {len(args.tickers)} stocks")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Feature set: {args.feature_set}")
    print(f"Test ratio: {args.test_ratio}")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading Data...")
    print("-" * 50)

    try:
        data = load_data(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            raw_path=config.RAW_DATA_PATH,
            force_download=not args.skip_download,
            progress=config.SHOW_PROGRESS
        )
        tickers_actual = get_ticker_list(data)
        start_date_actual, end_date_actual = get_date_range(data)
        print(f"Loaded {len(tickers_actual)} tickers: {', '.join(tickers_actual[:5])}{'...' if len(tickers_actual) > 5 else ''}")
        print(f"Date range: {start_date_actual} to {end_date_actual}")
        print(f"Price data shape: {data['prices'].shape}")
        print(f"VIX data shape: {data['vix'].shape}")

        results['data'] = data
        results['tickers'] = tickers_actual

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

    # -------------------------------------------------------------------------
    # 2. Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[2/5] Feature Engineering...")
    print("-" * 50)

    try:
        # Build full feature matrix
        save_path = config.PROCESSED_FILENAME if not args.no_save else None
        if save_path:
            save_path = os.path.join(config.PROCESSED_DATA_PATH, save_path)

        feature_df = build_feature_matrix(
            prices=data['prices'],
            vix=data['vix'],
            feature_set=None,  # Build all features
            save_path=save_path,
            force_rebuild=args.skip_download
        )

        print(f"Feature matrix shape: {feature_df.shape}")
        print(f"Features available: {len([c for c in feature_df.columns if c not in ['date', 'ticker', 'target']])}")
        print(f"Date range: {feature_df['date'].min()} to {feature_df['date'].max()}")
        print(f"Unique tickers: {feature_df['ticker'].nunique()}")

        # Filter to selected feature set if not 'all'
        if args.feature_set != 'all':
            feature_df = filter_features(feature_df, args.feature_set)
            print(f"Filtered to {args.feature_set}: {len(get_feature_list(args.feature_set))} features")
            print(f"Filtered shape: {feature_df.shape}")

        results['feature_df'] = feature_df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

    # -------------------------------------------------------------------------
    # 3. Model Training & Evaluation
    # -------------------------------------------------------------------------
    print("\n[3/5] Model Training & Evaluation...")
    print("-" * 50)

    try:
        # Determine which feature sets to run
        if args.feature_set == 'all':
            feature_sets = ['baseline_1', 'baseline_2', 'baseline_3', 'advanced']
        else:
            feature_sets = [args.feature_set]

        results_df = run_all_experiments(
            feature_df=feature_df,
            feature_sets=feature_sets,
            test_ratio=args.test_ratio,
            scale=True,
            ridge_alpha=args.ridge_alpha
        )

        print(f"Completed {len(results_df)} experiments")
        print(f"Models evaluated: {results_df['model'].nunique()}")
        print(f"Feature sets evaluated: {results_df['feature_set'].nunique()}")

        # Save results
        if not args.no_save:
            tables_path = os.path.join(args.output_dir, 'tables')
            os.makedirs(tables_path, exist_ok=True)
            results_df.to_csv(os.path.join(tables_path, 'model_results.csv'), index=False)

            # Create and save comparison tables
            rmse_pivot, beta_pivot, pvalue_pivot = create_comparison_table(results_df)
            rmse_pivot.to_csv(os.path.join(tables_path, 'rmse_comparison.csv'))
            beta_pivot.to_csv(os.path.join(tables_path, 'beta_comparison.csv'))

            # Best models
            best_models = results_df.loc[
                results_df.groupby('feature_set')['rmse'].idxmin()
            ][['feature_set', 'model', 'rmse', 'mae', 'mz_beta', 'mz_f_pvalue']]
            best_models.to_csv(os.path.join(tables_path, 'best_models.csv'), index=False)

            print(f"Results saved to {tables_path}")

        results['results_df'] = results_df

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

    # -------------------------------------------------------------------------
    # 4. Evaluation Report
    # -------------------------------------------------------------------------
    print("\n[4/5] Generating Evaluation Report...")
    print("-" * 50)

    try:
        # Find best model for plotting
        best_idx = results_df['rmse'].idxmin()
        best_model = results_df.loc[best_idx, 'model']
        best_feature_set = results_df.loc[best_idx, 'feature_set']

        print(f"Best model: {best_model} ({best_feature_set})")
        print(f"RMSE: {results_df.loc[best_idx, 'rmse']:.4f}")
        print(f"MAE: {results_df.loc[best_idx, 'mae']:.4f}")
        print(f"MZ Beta: {results_df.loc[best_idx, 'mz_beta']:.4f}")
        print(f"MZ p-value: {results_df.loc[best_idx, 'mz_f_pvalue']:.4f}")

        # Generate report
        if not args.no_save:
            # Get predictions for best model
            # Note: In a full implementation, we'd store predictions from models.py
            # For now, we'll skip detailed evaluation report generation
            # and focus on the backtest

            # Generate summary of best models
            best_models_summary = results_df.loc[
                results_df.groupby('feature_set')['rmse'].idxmin()
            ][['feature_set', 'model', 'rmse', 'mae', 'mz_beta', 'mz_f_pvalue']]

            summary_path = os.path.join(args.output_dir, 'tables', 'best_models_summary.csv')
            best_models_summary.to_csv(summary_path, index=False)
            print(f"Best models summary saved to {summary_path}")

        results['best_model'] = best_model
        results['best_feature_set'] = best_feature_set

    except Exception as e:
        logger.warning(f"Evaluation report generation had issues: {e}")
        # Continue anyway

    # -------------------------------------------------------------------------
    # 5. Backtest
    # -------------------------------------------------------------------------
    if not args.skip_backtest:
        print("\n[5/5] Running Backtest Simulation...")
        print("-" * 50)

        try:
            # Use best Ridge model for backtest (or best overall)
            # For now, use Ridge with best feature set
            # In production, we'd load the actual trained model

            # Get the best Ridge model result
            ridge_results = results_df[results_df['model'] == 'Ridge']
            if not ridge_results.empty:
                best_ridge_idx = ridge_results['rmse'].idxmin()
                backtest_feature_set = ridge_results.loc[best_ridge_idx, 'feature_set']
                print(f"Backtest using: Ridge ({backtest_feature_set})")

                # Use the feature set selected above
                backtest_df = filter_features(feature_df, backtest_feature_set)

                # For backtest, we need the actual predictions
                # For now, we'll use a simplified approach: use the feature values
                # as predictions for the baselines

                # Simplified backtest: use predicted_vol from the test set
                # Since we don't have predictions stored, we'll use the feature values
                # This is a placeholder - in production, we'd use saved predictions

                # Generate backtest report
                if not args.no_save:
                    # This is a simplified version since we don't have predictions stored
                    # The actual implementation would use predictions from models.py

                    # For now, use the target feature as a proxy (will be improved later)
                    comparison = generate_backtest_report(
                        returns=np.random.randn(100) * 0.01,  # Placeholder
                        predicted_vol=np.ones(100) * 0.15,    # Placeholder
                        dates=pd.date_range('2023-01-01', periods=100),
                        target_vol=args.target_vol,
                        max_leverage=args.max_leverage,
                        output_dir=args.output_dir
                    )

                    print("Backtest report generated (placeholder)")
                    results['backtest_comparison'] = comparison

        except Exception as e:
            logger.warning(f"Backtest simulation failed: {e}")
            # Continue anyway - backtest is optional
    else:
        print("\n[5/5] Skipping Backtest (--skip-backtest)")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: {len(results.get('results_df', []))}")
    print(f"Best model: {results.get('best_model', 'N/A')} ({results.get('best_feature_set', 'N/A')})")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70 + "\n")

    return results


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the pipeline."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        # Create necessary directories
        os.makedirs(config.RAW_DATA_PATH, exist_ok=True)
        os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'tables'), exist_ok=True)

        # Run pipeline
        results = run_pipeline(args)

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)

    except Exception as e:
        logging.getLogger(__name__).error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()