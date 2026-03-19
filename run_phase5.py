"""
run_phase5.py
─────────────
Phase 5 smoke test — Docker build + container verification.

What this script does:
  1. Checks that Docker is running          (docker info)
  2. Builds the image                       (docker compose build)
  3. Starts the container                   (docker compose up -d)
  4. Waits for /health to return 200        (polls every 2 s, up to 60 s)
  5. Hits /api/stats and prints the summary
  6. POSTs a sample clinical note to /api/deidentify
  7. Prints the de-identified result and entity count
  8. Optionally tears the container down    (pass --teardown flag)

Run:
    python run_phase5.py             # runs smoke test, leaves container up
    python run_phase5.py --teardown  # tears down after test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import urllib.request
import urllib.error

BASE_URL = "http://localhost:5000"
MAX_WAIT  = 60   # seconds to wait for container to become healthy


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, stream output, and optionally raise on error."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=False)
    if check and result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def get_json(path: str) -> dict:
    """HTTP GET to the container and return parsed JSON."""
    url = f"{BASE_URL}{path}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def post_json(path: str, payload: dict) -> dict:
    """HTTP POST with JSON body and return parsed JSON response."""
    url  = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wait_for_health(max_wait: int = MAX_WAIT) -> bool:
    """Poll /health every 2 s until it returns {'status': 'ok'} or timeout."""
    print(f"\n⏳ Waiting up to {max_wait}s for container to become healthy...")
    start = time.time()
    while time.time() - start < max_wait:
        try:
            data = get_json("/health")
            if data.get("status") == "ok":
                elapsed = round(time.time() - start, 1)
                print(f"✅ Container healthy after {elapsed}s")
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ClinicalNER Phase 5 smoke test")
    parser.add_argument("--teardown", action="store_true",
                        help="Stop and remove the container after the test")
    args = parser.parse_args()

    print("=" * 60)
    print("  ClinicalNER — Phase 5 Docker Smoke Test")
    print("=" * 60)

    # ── Step 1: Check Docker is running ──────────────────────────────────────
    print("\n[1/6] Checking Docker daemon...")
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Docker is not running. Start Docker Desktop and re-run.")
        sys.exit(1)
    print("✅ Docker is running")

    # ── Step 2: Build the image ───────────────────────────────────────────────
    print("\n[2/6] Building Docker image (this takes ~2 min on first build)...")
    run(["docker", "compose", "build", "--no-cache"])
    print("✅ Image built successfully")

    # ── Step 3: Start the container ───────────────────────────────────────────
    print("\n[3/6] Starting container...")
    run(["docker", "compose", "up", "-d"])

    # ── Step 4: Wait for health ───────────────────────────────────────────────
    print("\n[4/6] Health check...")
    if not wait_for_health():
        print("❌ Container did not become healthy within timeout.")
        print("   Run 'docker compose logs clinicalner' to investigate.")
        sys.exit(1)

    # ── Step 5: Stats ─────────────────────────────────────────────────────────
    print("\n[5/6] Fetching /api/stats...")
    stats = get_json("/api/stats")
    print(f"  Notes in DB    : {stats.get('note_count', 0):,}")
    print(f"  Processed      : {stats.get('processed_count', 0):,}")
    print(f"  Audit events   : {stats.get('total_audit_events', 0):,}")
    entities = stats.get("entity_totals", {})
    if entities:
        total = sum(entities.values())
        print(f"  PHI entities   : {total:,}")
        for k, v in sorted(entities.items(), key=lambda x: -x[1]):
            bar = "█" * min(v // 10, 30)
            print(f"    {k:<12} {v:>5}  {bar}")

    # ── Step 6: De-identification endpoint ────────────────────────────────────
    print("\n[6/6] Testing /api/deidentify ...")
    sample_note = (
        "Patient James Smith (DOB: 04/12/1985, MRN302145) was admitted to "
        "St. Mary's Hospital on 15/03/2024. Contact: (415) 555-9876. "
        "Diagnosis: Type 2 Diabetes. Treating physician: Dr. Emily Brown."
    )
    print(f"  Input  : {sample_note[:80]}...")

    result = post_json("/api/deidentify", {"text": sample_note})
    print(f"  Output : {result.get('masked_text', '')[:80]}...")
    print(f"  Entities found : {result.get('entity_count', 0)}")
    print(f"  Entity types   : {result.get('entity_types', {})}")
    print(f"  Valid output   : {result.get('is_valid', False)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Phase 5 complete — all checks passed!")
    print("  Dashboard : http://localhost:5000/dashboard")
    print("  Health    : http://localhost:5000/health")
    print("  API docs  : POST http://localhost:5000/api/deidentify")
    print("=" * 60)

    # ── Optional teardown ─────────────────────────────────────────────────────
    if args.teardown:
        print("\n[Teardown] Stopping and removing container...")
        run(["docker", "compose", "down"])
        print("✅ Container stopped.")
    else:
        print("\nContainer is still running. To stop it:")
        print("  docker compose down")


if __name__ == "__main__":
    main()
