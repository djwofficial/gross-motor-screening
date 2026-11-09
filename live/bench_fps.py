"""Quick check: does RTMPose run fast enough on the live laptop? Owner: Revallo. TODO (Nov 10).

Run it once on Freya's laptop (the live laptop), with a person in the picture:
    python -m live.bench_fps --device cuda --mode balanced --seconds 30

Print: GPU, device, mode, frames, mean fps. Pass: CUDA is used and fps is 15 or more (target 30).
If it fails, tell Daryn the same day: the fallback is MediaPipe.
"""


def main():
    raise NotImplementedError("Revallo, Nov 10: live/bench_fps.py")


if __name__ == "__main__":
    main()
