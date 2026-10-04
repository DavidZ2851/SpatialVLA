"""Serve a SpatialVLA LoRA checkpoint (fine-tuned on MolmoSpaces) over openpi's websocket protocol.

MolmoSpaces' eval client (PI_Policy, action_mode="spatialvla") connects with openpi_client's
WebsocketClientPolicy. Each request is one observation in the training layout of
data/molmo_dataset.py:

    image       (224, 224, 3) uint8    exterior view, already resized like the training frames
    intrinsic   (3, 3)        float32  that view's K for the 224x224 image (feeds Ego3D)
    prompt      str                    lower-cased task description

and the reply is {"actions": (4, 7)}: per-step [dxyz, drpy (euler xyz, left-multiplied), gripper
(1 = open)] in the robot base frame, unnormalized with the run's statistics (unnorm key
molmo_franka/1.0.0).

    python scripts/molmo/spatialvla_server.py --checkpoint <run>/checkpoint-20000 --port 8090
"""

import argparse
import asyncio
import functools
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import msgpack  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import websockets  # noqa: E402
import websockets.asyncio.server  # noqa: E402
from peft import PeftModel  # noqa: E402
from PIL import Image  # noqa: E402

from model import SpatialVLAConfig, SpatialVLAForConditionalGeneration, SpatialVLAProcessor  # noqa: E402

log = logging.getLogger("spatialvla_server")


# --- openpi_client.msgpack_numpy wire format ---------------------------------------------------
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class SpatialVLAPolicy:
    def __init__(self, args):
        self.args = args
        ckpt = Path(args.checkpoint)
        base = args.base or json_base(ckpt)
        # The checkpoint's processor carries the run's statistics (unnorm key) and action config.
        self.processor = SpatialVLAProcessor.from_pretrained(str(ckpt), local_files_only=True)
        config = SpatialVLAConfig.from_pretrained(base, torch_dtype=torch.bfloat16, local_files_only=True)
        model = SpatialVLAForConditionalGeneration.from_pretrained(
            base, config=config, torch_dtype=torch.bfloat16, local_files_only=True
        )
        model = PeftModel.from_pretrained(model, str(ckpt)).merge_and_unload()
        model.action_token_begin_idx = model.config.action_token_begin_idx = (
            self.processor.action_tokenizer.action_token_begin_idx
        )
        self.model = model.eval().to(args.device)
        log.info(f"loaded {ckpt} on {base}; unnorm key {args.unnorm_key} "
                 f"q01 {self.processor.statistics[args.unnorm_key]['action']['q01']}")

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        image = Image.fromarray(np.asarray(obs["image"], np.uint8))
        inputs = self.processor(
            images=[image], text=str(obs["prompt"]), unnorm_key=self.args.unnorm_key,
            return_tensors="pt", do_normalize=False,  # as training: Zoe and SigLIP normalize inside
        )
        inputs["intrinsic"] = torch.from_numpy(np.asarray(obs["intrinsic"], np.float32))[None]
        out = self.model.predict_action(inputs)
        actions = self.processor.decode_actions(out, unnorm_key=self.args.unnorm_key)["actions"]
        return {"actions": np.asarray(actions, np.float32)}


def json_base(ckpt: Path) -> str:
    import json

    return json.load(open(ckpt / "adapter_config.json"))["base_model_name_or_path"]


async def serve(policy: SpatialVLAPolicy, host: str, port: int):
    async def handler(ws):
        log.info(f"connection from {ws.remote_address}")
        await ws.send(packb({"model": "spatialvla"}))
        while True:
            try:
                obs = unpackb(await ws.recv())
                t = time.monotonic()
                out = policy.infer(obs)
                out["server_timing"] = {"infer_ms": (time.monotonic() - t) * 1000}
                await ws.send(packb(out))
            except websockets.ConnectionClosed:
                log.info(f"connection from {ws.remote_address} closed")
                break

    async with websockets.asyncio.server.serve(handler, host, port, compression=None, max_size=None):
        log.info(f"serving on {host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="a LoRA checkpoint-<step> dir")
    ap.add_argument("--base", default=None, help="pretrained model dir (default: the adapter's base path)")
    ap.add_argument("--unnorm_key", default="molmo_franka/1.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = ap.parse_args()
    asyncio.run(serve(SpatialVLAPolicy(a), a.host, a.port))
