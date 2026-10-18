#!/usr/bin/env python3
"""Publish the trained model to Hugging Face.

    pip install -U "huggingface_hub[hf_xet]"
    export HF_TOKEN=hf_...            # a *write* token
    python3 scripts/push_to_hf.py     # uploads artifacts/ + model-card/README.md
    python3 scripts/push_to_hf.py --dry-run

Hugging Face rejects binary files sent inline in a commit (they must go through Xet/LFS), so this
uses the official `huggingface_hub` client. It creates the model repo if needed and uploads
model.safetensors, config.json and the model card in a single commit.
"""
import argparse
import os
import sys

DEFAULT_REPO = "Max-imumlikelihood/cqlz-gym-policy"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="namespace/name of the model repo")
    ap.add_argument("--dir", default="artifacts", help="directory with model.safetensors and config.json")
    ap.add_argument("--card", default="model-card/README.md", help="model card, uploaded as README.md")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check inputs and the token, upload nothing")
    args = ap.parse_args()

    files = {
        "model.safetensors": os.path.join(args.dir, "model.safetensors"),
        "config.json": os.path.join(args.dir, "config.json"),
        "README.md": args.card,
    }
    for name, path in files.items():
        if not os.path.isfile(path):
            sys.exit(f"error: missing {path} (needed as {name})")

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        sys.exit('error: pip install -U "huggingface_hub[hf_xet]"  (then re-run)')

    token = os.environ.get("HF_TOKEN")
    if not token:
        if args.dry_run:
            print("dry run: inputs found; HF_TOKEN not set (needed for a real upload)")
            return
        sys.exit("error: set HF_TOKEN to a Hugging Face token with write access")

    api = HfApi(token=token)
    who = api.whoami()
    namespace = args.repo.split("/")[0]
    orgs = [o.get("name") for o in who.get("orgs", [])]
    if namespace != who["name"] and namespace not in orgs:
        sys.exit(f"error: this token belongs to '{who['name']}', not '{namespace}'; use --repo {who['name']}/<name>")
    if args.dry_run:
        print(f"dry run: token ok for '{who['name']}', would upload {', '.join(files)} to {args.repo}")
        return

    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.create_commit(
        args.repo,
        operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=p) for n, p in files.items()],
        commit_message="Upload cqlz-gym policy",
    )
    print(f"uploaded: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
