"""Measure RTMPose fps on this laptop. Owner: Louise. TODO (Nov 10).

Planned use (each NVIDIA laptop owner runs it, Louise collects the table):
    python -m live.bench_fps --device cuda --mode balanced --seconds 30

Print: laptop name, GPU, device, mode, frames, mean fps, and fps with video saving on.
The live laptop is picked from these numbers (target 30 fps, minimum 15).
"""


def main():
    raise NotImplementedError("Louise, Nov 10: live/bench_fps.py")


if __name__ == "__main__":
    main()
