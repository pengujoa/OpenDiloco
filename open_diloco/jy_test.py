import os, socket, time
import torch
import torch.nn as nn
from hivemind import DHT
from hivemind.utils import get_dht_time

RUN_ID = "OpenDiLoCo"

# The specific peer address you provided, correctly formatted
INITIAL_PEERS = [
    "/ip4/163.152.51.44/tcp/30001/p2p/QmSMekbSaNk5sTL4LfnoMxnCBdVKckSXMAo3wMh4NHKbGz",
]

GPU = int(os.getenv("LOCAL_RANK", "0"))

PUBLISH_INTERVAL = 10.0
TTL = 30.0

DUMMY_BATCH_SIZE = 256
DUMMY_INPUT_DIM = 4096
DUMMY_HIDDEN_DIM = 8192
DUMMY_OUTPUT_DIM = 4096

WARMUP_STEPS = 20
BENCHMARK_STEPS = 500

def measure_steps_per_second(dev):
    model = nn.Sequential(
        nn.Linear(DUMMY_INPUT_DIM, DUMMY_HIDDEN_DIM),
        nn.ReLU(),
        nn.Linear(DUMMY_HIDDEN_DIM, DUMMY_OUTPUT_DIM),
    ).to(dev, dtype=torch.float16)
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    inputs = torch.randn(DUMMY_BATCH_SIZE, DUMMY_INPUT_DIM, device=dev, dtype=torch.float16)
    targets = torch.randint(0, DUMMY_OUTPUT_DIM, (DUMMY_BATCH_SIZE,), device=dev)

    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(dev)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(BENCHMARK_STEPS):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    end.record()
    
    torch.cuda.synchronize(dev)
    
    elapsed_ms = start.elapsed_time(end)
    elapsed_sec = elapsed_ms / 1000
    steps_per_sec = BENCHMARK_STEPS / elapsed_sec
    
    return steps_per_sec

def main():
    torch.cuda.set_device(GPU)
    gpu_name = torch.cuda.get_device_name(GPU)
    
    dht = DHT(initial_peers=INITIAL_PEERS, start=True, client_mode=False)

    worker_id = f"{socket.gethostname()}-pid{os.getpid()}-gpu{GPU}"
    key = f"{RUN_ID}:speed"

    while True:
        steps_per_sec = measure_steps_per_second(GPU)
        
        now = get_dht_time()
        exp = now + TTL

        payload = {
            "steps_per_sec": float(steps_per_sec), 
            "ts": now, 
            "host": socket.gethostname(), 
            "gpu_id": GPU,
            "gpu_name": gpu_name
        }
        
        ok = dht.store(key=key, subkey=worker_id, value=payload, expiration_time=exp)
        

        print(f"[publish] {worker_id} ({gpu_name}): {steps_per_sec:.2f} steps/sec ({'ok' if ok else 'fail'})")
        
        time.sleep(PUBLISH_INTERVAL)

if __name__ == "__main__":
    main()