"""Experimental group-only tournament scenario; simplified bracket, no market comparison."""
import argparse
from pathlib import Path

from data_contract import load_results
from strength import StrengthModel
from tournament import Tournament


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results.csv"))
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="Complete 72-match group-stage-only CSV, with result-schema columns")
    parser.add_argument("--cutoff", default="2026-06-11")
    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("win_probabilities.csv"))
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {args.data.resolve(), args.fixtures.resolve()}:
            raise ValueError("Output must not overwrite data or fixtures")
        fixtures = load_results(args.fixtures)
        model = StrengthModel().fit(args.data, cutoff=args.cutoff)
        result = Tournament(model, fixtures).monte_carlo(args.simulations, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print("Experimental scenario: simplified seeding/tiebreaks, independent Poisson matches.")
        print(result.head(15).to_string(index=False))
        return 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Tournament simulation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
