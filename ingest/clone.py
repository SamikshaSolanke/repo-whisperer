import argparse
from pathlib import Path
import git
import config


def clone(repo: str) -> Path:
    dest = Path(config.REPOS_DIR) / repo.split("/")[-1]
    if dest.exists():
        print(f"{dest} already exists, reusing it")
    else:
        git.Repo.clone_from(f"https://github.com/{repo}.git", dest, depth=1)
    print(f"{repo} @ {git.Repo(dest).head.commit.hexsha}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="e.g. psf/requests")
    clone(ap.parse_args().repo)