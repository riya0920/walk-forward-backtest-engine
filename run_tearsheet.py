"""Write docs/TEARSHEET.md and data/blotter.csv from a real run.

    python run_tearsheet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.backtest import buy_and_hold, run
from engine.strategies import make_synthetic_prices, momentum
from engine.tearsheet import blotter, round_trips, tearsheet


def main() -> None:
    prices = make_synthetic_prices()
    res = run(prices, momentum(20))
    bench = buy_and_hold(prices)

    doc = tearsheet(res, bench, name="momentum(20), 8bps round trip, synthetic walk")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "TEARSHEET.md").write_text(doc, encoding="utf-8")

    b = blotter(res)
    (ROOT / "data").mkdir(exist_ok=True)
    b.to_csv(ROOT / "data" / "blotter.csv", index=False)
    rt = round_trips(res)
    rt.to_csv(ROOT / "data" / "round_trips.csv", index=False)

    print(doc)
    print()
    print("wrote docs/TEARSHEET.md, data/blotter.csv ({} fills), "
          "data/round_trips.csv ({} trades)".format(len(b), len(rt)))
    print()
    print("The strategy is test cargo, as everywhere else in this repo. What is")
    print("being demonstrated is that the headline table can be reassembled from")
    print("the fills that produced it.")


if __name__ == "__main__":
    main()
