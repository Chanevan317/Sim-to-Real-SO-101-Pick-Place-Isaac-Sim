import socket
import pickle
import struct
import numpy as np


class PolicyClient:
    def __init__(self, host="127.0.0.1", port=9999):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def _send(self, data: bytes):
        self.sock.sendall(struct.pack(">I", len(data)) + data)

    def _recv(self) -> bytes:
        raw_len = self._recvall(4)
        msg_len = struct.unpack(">I", raw_len)[0]
        return self._recvall(msg_len)

    def _recvall(self, n):
        data = b""
        while len(data) < n:
            packet = self.sock.recv(n - len(data))
            if not packet:
                raise ConnectionError("Policy server closed connection.")
            data += packet
        return data

    def get_action(self, state: np.ndarray, image_ego: np.ndarray, image_external: np.ndarray) -> np.ndarray:
        # Requests still use pickle (dict of small float32 arrays) -- this
        # direction has been working fine. Only the numpy-array RESPONSE
        # needs the version-agnostic path below.
        payload = pickle.dumps({
            "state": state.astype(np.float32),
            "image_ego": image_ego.astype(np.float32),
            "image_external": image_external.astype(np.float32),
        })
        self._send(payload)

        raw = self._recv()
        # Response format: 4-byte ndim, then ndim * 4-byte shape dims,
        # then raw float32 bytes -- no pickle, no numpy-internal references.
        ndim = struct.unpack(">I", raw[:4])[0]
        offset = 4
        shape = struct.unpack(f">{ndim}I", raw[offset:offset + 4 * ndim])
        offset += 4 * ndim
        action = np.frombuffer(raw[offset:], dtype=np.float32).reshape(shape).copy()
        return action

    def close(self):
        self.sock.close()