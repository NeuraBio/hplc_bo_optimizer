# generate_rt_files.py
import csv
import random
from pathlib import Path


def generate_rt_file(output_path: Path, num_peaks: int = 4, noise: float = 0.1):
    """Generate a realistic RT CSV with slight randomness."""
    base_rts = [4.0, 5.0, 6.0, 7.0]  # Baseline RTs (modify as needed)
    rts = [round(base + random.uniform(-noise, noise), 2) for base in base_rts[:num_peaks]]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["RT"])  # Header
        for rt in sorted(rts):  # Ensure RTs are ordered
            writer.writerow([rt])


# Generate 20 files
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

for i in range(1, 21):
    generate_rt_file(output_dir / f"rt_{i}.csv")
