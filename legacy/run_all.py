"""Driver. Enforces the execution order of plan Sec. 7.

    python run_all.py

Stages, in order (each writes to out/ and is resumable):
  0. tests           -- abort on any failure
  1. gauss run       -- cheap n=0 control; catches truncation/noise bugs
  2. moliere run     -- the expensive 2e6-event sample
  3. branch A        -- eps_M(p) + fit
  4. branch B        -- images, metrics, artifact maps
  5. correction      -- corrected image metrics (inside branch B)
"""

import os
import subprocess
import sys

from config import OUT_DIR


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("== stage 0: validation ==")
    if subprocess.call([sys.executable, "tests.py"]) != 0:
        sys.exit("validation failed -- fix before running the production sample")

    print("== stage 1: Gaussian control run (n=0) ==")
    from . import simulate

    simulate.run(mode="gauss")

    print("== stage 2: Moliere production run ==")
    simulate.run(mode="moliere")

    print("== stage 3: Branch A ==")
    from . import branch_a

    dfa = branch_a.branch_a()
    print(dfa.to_string(index=False))
    a, b = branch_a.fit_eps_M(dfa)
    print(f"eps_M(p) = {a:+.5f} {b:+.5f} * ln p")
    for m in branch_a.sanity(dfa):
        print(m)

    print("== stage 4/5: Branch B + correction ==")
    from . import branch_b

    res, summary = branch_b.run_branch_b()
    print(res.to_string(index=False))
    print(summary)
    branch_b.run_per_setting()

    print("done. outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
