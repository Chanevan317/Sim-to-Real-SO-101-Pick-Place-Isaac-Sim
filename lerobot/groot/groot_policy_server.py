"""
GR00T N1.7 policy inference server.

Drop-in replacement for ../policy_server.py: same socket protocol, so
source/sim_to_real_so101/scripts/lerobot_eval.py works unchanged against it.

Differences from the ACT server:
    - Sends the language instruction ("task") to the policy.
    - Predicts a full action chunk, post-processes it as a whole and replays
      it step by step. GrootPolicy.select_action() refuses relative-action
      checkpoints, because each relative chunk must be decoded against the
      state it was predicted from.
    - Resets policy/processor state when the client flags a new episode.

Usage (lerobot env):
    python groot_policy_server.py \
        --checkpoint ../train/outputs/so101_groot/checkpoints/020000/pretrained_model
"""

import argparse
import os
import pickle
import socket
import sys
from collections import deque
from pathlib import Path

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy_server import encode_ndarray, recv_msg, send_msg  # noqa: E402

# HF_TOKEN for the gated nvidia/Cosmos-Reason2-2B repo: the Qwen2VL processor
# is only built on the first request, so without this the server looks fine
# at startup and then dies on the sim's first observation. Load the same
# gitignored .env train_groot.sh uses, without overriding a token already
# exported in the shell.
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

DEFAULT_TASK = "pick up the cube and place it on the plate"


def main():
    parser = argparse.ArgumentParser(description="GR00T N1.7 policy inference server.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a pretrained_model dir, e.g. "
             "train/outputs/so101_groot/checkpoints/020000/pretrained_model",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        help="Fallback instruction when the client does not send one. Must match the dataset task.",
    )
    parser.add_argument(
        "--execute_steps",
        type=int,
        default=None,
        help="Execute only the first N actions of each chunk before re-planning "
             "(default: the whole chunk, i.e. n_action_steps).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = GrootPolicy.from_pretrained(args.checkpoint)
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Policy server listening on {args.host}:{args.port}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Chunk: n_action_steps={policy.config.n_action_steps}, execute_steps={args.execute_steps or 'all'}")

    conn, addr = server.accept()
    print(f"Connected: {addr}")

    action_queue = deque()

    with torch.inference_mode():
        request_count = 0
        while True:
            raw = recv_msg(conn)
            if raw is None:
                print("Client disconnected.")
                break
            batch = pickle.loads(raw)
            request_count += 1

            if batch.get("reset", False):
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                action_queue.clear()
                print(f"[SERVER] Episode reset at request #{request_count}")

            if not action_queue:
                obs = {
                    "observation.state": torch.from_numpy(batch["state"]).float().unsqueeze(0).to(device),
                    "observation.images.ego": torch.from_numpy(batch["image_ego"]).float().unsqueeze(0).to(device),
                    "observation.images.external_D455": torch.from_numpy(batch["image_external"]).float().unsqueeze(0).to(device),
                    "task": batch.get("task") or args.task,
                    "robot_type": "so101",
                }
                obs = preprocessor(obs)
                chunk = policy.predict_action_chunk(obs)  # (1, n_action_steps, action_dim)
                chunk = postprocessor(chunk).squeeze(0)    # (n_action_steps, action_dim), real degrees
                if args.execute_steps is not None:
                    chunk = chunk[: args.execute_steps]
                action_queue.extend(chunk.float().cpu().numpy())

            action_np = action_queue.popleft()
            print(f"[SERVER] Response #{request_count} | action={action_np}")
            send_msg(conn, encode_ndarray(action_np))

    conn.close()
    server.close()


if __name__ == "__main__":
    main()
