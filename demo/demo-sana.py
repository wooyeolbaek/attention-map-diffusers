from run_attention_demo import main
import sys


if __name__ == "__main__":
    raise SystemExit(main(["--model", "sana", "--relations", "all", *sys.argv[1:]]))
