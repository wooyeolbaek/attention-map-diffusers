import sys

from run_attention_demo import main


if __name__ == "__main__":
    raise SystemExit(main(["--model", "cogvideox-2b", "--relations", "all", *sys.argv[1:]]))
