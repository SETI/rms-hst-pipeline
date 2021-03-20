import os
import sys
LOG_DIR = "lidvid_error_log.txt"

if __name__ == "__main__":
    with open(LOG_DIR, "r") as f:
        for line in f.readlines():
            msg = f"{line}, please refer to {LOG_DIR}"
            assert "to absent" not in line, msg
