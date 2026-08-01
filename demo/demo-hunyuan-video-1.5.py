import sys

from run_attention_demo import main


if __name__ == "__main__":
    raise SystemExit(main(["--model", "hunyuan-video-1.5", "--relations", "all", *sys.argv[1:]]))
