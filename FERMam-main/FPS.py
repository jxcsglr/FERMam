import torch
import time
import numpy as np
from models.emotion_hyp import pyramid_trans_expr

def benchmark(model, input_size=(1, 3, 224, 224), warmup=50, iters=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(iters):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    avg_latency = total_time / iters * 1000  # ms
    fps = input_size[0] / (total_time / iters)

    return avg_latency, fps


if __name__ == "__main__":
    # -------- model config --------
    num_classes = 7
    model_type = "large"   # small / base / large
    checkpoint_path = "/home/cdu-cs/jx/J/FERMam/FERMam-main/checkpoint/best_epoch.pth"  # 填你的 best ckpt

    model = pyramid_trans_expr(
        img_size=224,
        num_classes=num_classes,
        type=model_type
    )

    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # -------- Latency (batch=1) --------
    latency, fps1 = benchmark(
        model,
        input_size=(1, 3, 224, 224)
    )
    print(f"[Latency] Batch=1 | {latency:.2f} ms | FPS={fps1:.2f}")

    # -------- Throughput (batch=32) --------
    latency_b32, fps32 = benchmark(
        model,
        input_size=(32, 3, 224, 224)
    )
    print(f"[Throughput] Batch=32 | {latency_b32:.2f} ms | FPS={fps32:.2f}")
