import argparse
import json
from pathlib import Path

from qecsim.simulation import run


# Read a configuration and run or resume a portable Monte Carlo experiment.
def main():
    parser = argparse.ArgumentParser(description="Run fixed-sample QEC measurement simulations.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path(__file__).parent / "models")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config, args.output, args.models, args.workers)
    for cell in result["cells"]:
        metric = cell["metrics"].get("complete_instruction", {})
        text = f"{cell['mode']} p={cell['p']:g}: {cell['completed']}/{cell['shots']} shots, {metric.get('failures', 0)} failures"
        if cell["complete"]:
            text += f", LER={metric['rate']:.6g}, 95% CI [{metric['ci95_low']:.6g}, {metric['ci95_high']:.6g}]"
        print(text, flush=True)
    print(f"Saved {args.output / 'summary.json'} and summary.csv", flush=True)
    return 130 if result["interrupted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
