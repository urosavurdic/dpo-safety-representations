"""
Regression test for logs/release_gap_audit.md item 6 / section 5-adjacent
finding: `--dry-run --regenerate --with-probes` used to produce byte-
identical output to the same command without `--with-probes`, because
`main_run()` returned right after `describe_plan()` -- before ever
reaching the post-loop `if args.with_probes: compute_probes(...)` branch.
The per-stage plan table has no field for probes (they are a single
post-loop CPU-aggregation step, not a per-stage one), so nothing in the
dry-run's execution graph reflected the flag.

This test runs the real CLI twice (dry-run only, no model code executes)
and asserts the outputs now differ, and that the differing line correctly
reports probes as requested/not-requested. It intentionally does NOT
touch or re-assert anything about actual probe *computation* -- only
about the dry-run's textual report of the plan.
"""
import subprocess
import sys


def _run_dry(*extra_args):
    result = subprocess.run(
        [sys.executable, "-m", "src.analysis.v2_pipeline", "run",
         "--dry-run", "--regenerate", *extra_args],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"dry-run failed: {result.stderr}"
    return result.stdout


def test_with_probes_changes_dry_run_output():
    without = _run_dry()
    with_probes = _run_dry("--with-probes")
    assert without != with_probes, (
        "--with-probes must be visible in --dry-run output "
        "(see logs/release_gap_audit.md item 6)"
    )


def test_dry_run_reports_probes_will_run_when_requested():
    out = _run_dry("--with-probes")
    assert "probes WILL run" in out


def test_dry_run_reports_probes_will_not_run_when_not_requested():
    out = _run_dry()
    assert "probes will NOT run" in out


def test_dry_run_never_executes_model_code():
    out = _run_dry("--with-probes")
    assert "Dry run complete. No model code was executed." in out
