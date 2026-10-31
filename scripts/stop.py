"""Ask a running `train.py` to stop gracefully: it finishes the iteration in progress, saves a checkpoint, and
exits - the "button" for a training run you're not attached to (detached tmux/screen, a remote cloud box, or just
a different terminal). Ctrl-C in the training terminal itself does the same thing.

  python scripts/stop.py selfplay          # the same --name you trained with
"""
import argparse, os

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("name")
  ap.add_argument("--outDir", default="runs")
  args = ap.parse_args()

  path = os.path.join(args.outDir, args.name, "STOP")
  if not os.path.isdir(os.path.dirname(path)):
    raise SystemExit(f"no run directory {os.path.dirname(path)} - is {args.name!r} the --name it was trained with?")
  open(path, "w").close()
  print(f"wrote {path} - it will stop after its current iteration")

if __name__ == "__main__":
  main()
