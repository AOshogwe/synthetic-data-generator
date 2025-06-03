import argparse
from pipeline import SyntheticDataPipeline


def main():
    parser = argparse.ArgumentParser(description='Synthetic Data Generator')
    parser.add_argument('--input', required=True, help='Path to input data (directory or connection string)')
    parser.add_argument('--output', required=True, help='Path to output synthetic data')
    parser.add_argument('--format', default='csv', choices=['csv', 'json', 'parquet', 'sqlite', 'excel', 'zip'],
                        help='Output format type')
    parser.add_argument('--method', default='auto', choices=['auto', 'ctgan', 'tvae', 'copula'],
                        help='Generation method to use')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run without interactive prompts')
    parser.add_argument('--print-evaluation', action='store_true',
                        help='Print detailed evaluation comparison between original and synthetic data')
    parser.add_argument('--evaluation-only', action='store_true',
                        help='Only print evaluation without generating new data')

    args = parser.parse_args()

    # Create pipeline
    pipeline = SyntheticDataPipeline(args.config)

    # Run pipeline
    success = pipeline.run_pipeline(
        input_path=args.input,
        output_path=args.output,
        format_type=args.format,
        generation_method=args.method,
        interactive=not args.non_interactive
    )

    if success:
        print("Synthetic data generation completed successfully")
    else:
        print("Synthetic data generation failed")
        return 1

    return 0


if __name__ == "__main__":
    main()