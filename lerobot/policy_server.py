import socket
import pickle
import struct
import torch
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline

CHECKPOINT_PATH = "train/outputs/so101_act/checkpoints/030000/pretrained_model"
HOST, PORT = "127.0.0.1", 9999

def recv_msg(conn):
    raw_len = recvall(conn, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    return recvall(conn, msg_len)

def recvall(conn, n):
    data = b""
    while len(data) < n:
        packet = conn.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def send_msg(conn, data):
    conn.sendall(struct.pack(">I", len(data)) + data)

def encode_ndarray(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    header = struct.pack(">I", arr.ndim) + struct.pack(f">{arr.ndim}I", *arr.shape)
    return header + arr.tobytes()

def main():
    policy = ACTPolicy.from_pretrained(CHECKPOINT_PATH)
    policy.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT_PATH, config_filename="policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        CHECKPOINT_PATH, config_filename="policy_postprocessor.json"
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Policy server listening on {HOST}:{PORT}")

    conn, addr = server.accept()
    print(f"Connected: {addr}")

    with torch.inference_mode():
        request_count = 0
        while True:
            raw = recv_msg(conn)
            if raw is None:
                print("Client disconnected.")
                break
            batch = pickle.loads(raw)

            request_count += 1

            torch_batch = {
                "observation.state": torch.from_numpy(batch["state"]).float().unsqueeze(0).to(device),
                "observation.images.ego": torch.from_numpy(batch["image_ego"]).float().unsqueeze(0).to(device),
                "observation.images.external_D455": torch.from_numpy(batch["image_external"]).float().unsqueeze(0).to(device),
            }

            # print("preprocessor steps:", preprocessor.steps)
            # print("pre-preprocessor image stats:", torch_batch["observation.images.ego"].min().item(),
            #     torch_batch["observation.images.ego"].max().item())
            
            processed_batch = preprocessor(torch_batch)
            
            # print("post-preprocessor image stats:", processed_batch["observation.images.ego"].min().item(),
            #     processed_batch["observation.images.ego"].max().item())
            # print("preprocessor output keys:", processed_batch.keys())

            action = policy.select_action(processed_batch)
            action = postprocessor({"action": action})

            # print("postprocessor output type:", type(action))
            if isinstance(action, dict):
                # print("postprocessor output keys:", action.keys())
                action = action["action"]  # or whatever key it actually returns

            action_np = action.squeeze(0).cpu().numpy()

            print(f"[SERVER] Response #{request_count} | action={action_np}")

            send_msg(conn, encode_ndarray(action_np))

    conn.close()
    server.close()

if __name__ == "__main__":
    main()