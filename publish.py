import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger()

DATA_JSON = "data.json"

remote = "gage"
bucket = "gage-boards"
board_key = "chatbot-arena"
test_credentials = False

target_obj = f"{remote}:{bucket}/{board_key}.json"


def main():
    _check_rclone()
    _check_credentials()
    board = _generate_board()
    _save_board(board)
    _publish_board()


def _check_rclone():
    try:
        out = subprocess.check_output(
            "rclone --version",
            shell=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        raise SystemExit(
            "Cannot find rclone\n"
            "Confirm that it's installed and available on the system path.\n"
            "Refer to https://rclone.org/install/ for help."
        )
    else:
        log.info("Found %s", out.split("\n")[0])


def _check_credentials():
    log.info("Checking remote credentials")
    try:
        subprocess.run(
            f"rclone touch {target_obj}",
            shell=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise SystemExit(
            "There's a problem with the remote configuration. Confirm that "
            "your environment is setup with the required credentials and that "
            f"you have permission to write to {target_obj}"
        )
    else:
        if test_credentials:
            log.info("Credentials are OK")
            raise SystemExit(0)


def _generate_board():
    log.info("Generating board using board.json")
    return subprocess.check_output(
        "gage board --config board.json --json",
        shell=True,
        text=True,
    )


def _save_board(board):
    log.info("Saving board to %s", DATA_JSON)
    with open(DATA_JSON, "w") as f:
        f.write(board)


def _publish_board():
    log.info("Publishing board to %s", target_obj)
    try:
        subprocess.run(
            f"rclone copyto {DATA_JSON} {target_obj}",
            shell=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        log.error(
            "There was an error publishing the board. Refer to the "
            "output above for details."
        )
        raise SystemExit(e.returncode)


if __name__ == "__main__":
    main()
