import pathlib
import subprocess
import sys
import json
from typing import Any, Optional


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CLI_PATH = ROOT_DIR / "scripts" / "cli_recommend.py"


def _prompt(msg: str, *, default: Optional[str] = None) -> str:
    while True:
        suffix = f" (default: {default})" if default is not None else ""
        val = input(f"{msg}{suffix}: ").strip()
        if val:
            return val
        if default is not None:
            return default


def _prompt_bool(msg: str, *, default: bool = True) -> bool:
    while True:
        default_str = "Y/n" if default else "y/N"
        val = input(f"{msg} ({default_str}, default={'Y' if default else 'N'}): ").strip().lower()
        if not val:
            return default
        if val in {"y", "yes", "true", "1"}:
            return True
        if val in {"n", "no", "false", "0"}:
            return False
        print("Please answer with y/yes or n/no.")


def _run_cli_json(cmd: list[str]) -> dict[str, Any]:
    """
    Run `cli_recommend.py` and parse its JSON output.

    Warnings printed to stderr are still forwarded to the terminal.
    """
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.stderr:
        # stderr often contains helpful warnings like "[warn] Unknown movieId_raw=..."
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"CLI returned non-zero exit code {proc.returncode}.")
    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise RuntimeError("CLI produced empty output.")
    try:
        return json.loads(stdout)
    except Exception as e:
        raise RuntimeError(f"Failed to parse CLI JSON output: {e}\nOutput was:\n{stdout}")


def _format_title(title: str) -> str:
    # Keep CLI-provided title verbatim; titles already include year in parentheses.
    return title.strip()


def _print_recommendations(payload: dict[str, Any], *, score_decimals: int = 4) -> None:
    dataset = payload.get("dataset", "")
    mode = payload.get("mode", "")
    top_k = payload.get("top_k", payload.get("recommendations", []).__len__())
    provided_pairs = payload.get("cold_start_provided_pairs", None)
    user_id_raw = payload.get("userId_raw", None)

    header = f"Top {top_k} recommendations for {mode.replace('_', ' ')} ({dataset})"
    if mode == "cold_start" and provided_pairs is not None:
        header += f" from {provided_pairs} provided ratings"
    if mode == "existing_user" and user_id_raw is not None:
        header += f" (userId={user_id_raw})"
    print(header)
    print("-" * len(header))

    recommendations = payload.get("recommendations", []) or []
    for rec in recommendations:
        rank = rec.get("rank", "")
        title = _format_title(str(rec.get("title", "")))
        score = rec.get("score", None)
        genres = str(rec.get("genres", "")) if rec.get("genres", None) is not None else ""
        movie_id_raw = rec.get("movieId_raw", None)

        score_disp = "n/a"
        if isinstance(score, (int, float)):
            score_disp = f"{float(score):.{score_decimals}f}"

        idx = f"{int(rank):2d}" if isinstance(rank, int) or (isinstance(rank, str) and rank.isdigit()) else str(rank)
        print(f"{idx}. {title}")
        if movie_id_raw is not None:
            print(f"    movieId: {movie_id_raw} | score: {score_disp}")
        else:
            print(f"    score: {score_disp}")
        if genres:
            print(f"    genres: {genres.replace('|', ', ')}")
        print()


def main() -> None:
    print("Movie recommendation shell")
    print("Type 'q' at a prompt to quit.\n")

    dataset = _prompt("Choose dataset", default="10m")
    while dataset not in {"1m", "10m"}:
        if dataset.lower() == "q":
            return
        dataset = _prompt("Choose dataset (1m or 10m)", default="10m")

    while True:
        mode = _prompt("Mode: (u) existing user, (c) cold-start", default="u").lower()
        if mode == "q":
            return
        if mode not in {"u", "c"}:
            print("Invalid mode. Use 'u' or 'c'.")
            continue

        top_k_s = _prompt("Top-K", default="10")
        if top_k_s.lower() == "q":
            return
        try:
            top_k = int(top_k_s)
        except ValueError:
            print("Top-K must be an integer.")
            continue

        exclude_rated = _prompt_bool("Exclude already-rated movies", default=True)
        output_format = _prompt("Output format: text or json", default="text").lower()
        if output_format == "q":
            return
        if output_format not in {"text", "json"}:
            print("Output format must be 'text' or 'json'.")
            continue

        cmd: list[str] = [sys.executable, str(CLI_PATH), "--dataset", dataset, "--top-k", str(top_k)]
        # We call the underlying CLI in JSON mode so we can render pretty output.
        cmd += ["--format", "json", "--score-decimals", "4"]

        # By default `cli_recommend.py` excludes seen items; to *include* seen items we pass `--include-seen`.
        if not exclude_rated:
            cmd += ["--include-seen"]

        try:
            if mode == "u":
                user_id_s = _prompt("Existing MovieLens raw userId", default="1")
                if user_id_s.lower() == "q":
                    return
                user_id = int(user_id_s)
                cmd += ["--user", str(user_id)]
            else:
                ratings = _prompt(
                    "Cold-start ratings: enter movieId:rating,... or @file",
                    default="1:5,260:3.5,1193:4",
                )
                if ratings.lower() == "q":
                    return
                if ratings.startswith("@"):
                    ratings_path = pathlib.Path(ratings[1:]).expanduser().resolve()
                    cmd += ["--ratings-file", str(ratings_path)]
                else:
                    cmd += ["--ratings", ratings]

            payload = _run_cli_json(cmd)
            if output_format == "json":
                print(json.dumps(payload, indent=2))
            else:
                _print_recommendations(payload, score_decimals=4)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

