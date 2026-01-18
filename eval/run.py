import argparse
import json

from app.evaluation import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation.")
    parser.add_argument("--path", required=True, help="Path to eval JSON file.")
    args = parser.parse_args()

    summary = run_eval(args.path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
