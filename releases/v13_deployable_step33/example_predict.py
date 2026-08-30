from pathlib import Path
import argparse

from inference import V13Step33Predictor


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict backbone-amide 1H and 15N chemical shifts "
            "from a PDB or mmCIF structure."
        )
    )

    parser.add_argument("structure")
    parser.add_argument("--chain", required=True)
    parser.add_argument(
        "--release-dir",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument(
        "--output",
        default="hsqc_predictions.csv",
    )
    parser.add_argument(
        "--include-proline",
        action="store_true",
    )
    parser.add_argument(
        "--device",
        default=None,
    )

    args = parser.parse_args()

    predictor = V13Step33Predictor(
        args.release_dir,
        device=args.device,
    )

    predictions = predictor.predict_pdb(
        args.structure,
        chain_id=args.chain,
        include_proline=args.include_proline,
    )

    predictions.to_csv(args.output, index=False)

    print(predictions.head(20).to_string(index=False))
    print(f"Predicted rows: {len(predictions)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
