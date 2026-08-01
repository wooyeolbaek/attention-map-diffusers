import sys

from run_attention_demo import main


if __name__ == "__main__":
    raise SystemExit(main(["--model", "wan2.1-t2v-1.3b", "--relations", "all", *sys.argv[1:]]))
