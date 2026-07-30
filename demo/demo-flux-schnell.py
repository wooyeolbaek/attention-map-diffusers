from run_attention_demo import main
import sys


if __name__ == "__main__":
    raise SystemExit(main(["--model", "flux-schnell", "--relations", "all", *sys.argv[1:]]))
