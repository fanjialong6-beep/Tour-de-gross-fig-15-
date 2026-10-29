from pathlib import Path
import argparse

from qecsim.four_modes import MODES, make_four_model


# Build and save the four circuit fault models used by the simulator.
def main():
    parser = argparse.ArgumentParser(description="Build the current logical-measurement models.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    for mode in args.modes:
        model = make_four_model(mode)
        model.save(args.output / mode)
        print(f"{mode}: H={model.h.shape}, A={model.a.shape}, period={model.metadata['schedule_period']}", flush=True)


if __name__ == "__main__":
    main()
