# Phase 10 — Scaling & Resource Monitoring

## Goals

1. Test with **larger models** (if available)
2. Test with **bigger context windows** (1000+ tokens)
3. **Memory leak detection** with agent-based monitoring
4. Compare **Dense vs MoE** Mamba models (if available)

---

## Test Matrix

### Model Types

| Model | Architecture | Status | Notes |
|-------|--------------|--------|-------|
| `granite-4.0-h-tiny` | MoE + Hybrid (Mamba+Attn) | ✅ Tested | Current baseline |
| `Nemotron-3-Nano-4B-BF16` | Dense (Transformer) | ⚠️ No Mamba | Won't test - not Mamba |
| `granite-4.0-h-tiny-gguf` | Quantized MoE | ⚠️ Unsupported | GGUF not supported by SGLang |
| **Need: Larger Mamba model** | Pure Mamba or Dense Mamba | ❌ Not available | Download if exists |

**Finding:** We've only tested **MoE + Hybrid** so far. Need to test:
- Pure Dense Mamba (if any exist in the ecosystem)
- Larger parameter counts (7B+, 13B+)

### Context Window Testing

| Test | Input Size | Output Size | Turns | Goal |
|------|-----------|-------------|-------|------|
| Small (current) | ~50 tokens | ~50 tokens | 4 | ✅ Baseline |
| Medium | ~500 tokens | ~100 tokens | 5 | Test snapshot size growth |
| Large | ~2000 tokens | ~200 tokens | 3 | Test memory pressure |
| XL | ~8000 tokens | ~500 tokens | 2 | Push context limits |

---

## Memory Leak Detection Strategy

### Agent Team Composition

```python
# Monitor Agent (runs continuously)
metrics_agent = Agent(
    role="Resource Monitor",
    tools=[
        "nvidia-smi",           # GPU VRAM, utilization, temp
        "free",                 # System RAM
        "docker stats",         # Container stats (if running in container)
        "ps aux",               # Process memory
    ],
    poll_interval=5s,           # Check every 5 seconds
    log_to_file="phase-10-metrics.log"
)

# Load Agent (runs test scenarios)
load_agent = Agent(
    role="Load Generator",
    tools=["requests", "asyncio"],
    scenarios=[
        "multi_turn_conversation",
        "concurrent_users",
        "long_context_window",
    ]
)
```

### Metrics to Track

| Metric | Tool | Acceptable Range | Alert Threshold |
|--------|------|------------------|-----------------|
| GPU VRAM | nvidia-smi | Stable after warmup | Growth > 500MB/10min |
| GPU Utilization | nvidia-smi | 40-90% | >95% (risk of OOM) |
| System RAM | free | Stable | Growth > 1GB/10min |
| Container Memory | docker stats | Stable | Growth > 1GB/10min |
| Process RSS | ps | Stable | Growth > 500MB/10min |
| File descriptors | lsof | Stable | Growth > 100/10min |
| Disk I/O | iostat | Bursty only | Sustained high write |

### Memory Leak Test Scenarios

1. **Baseline Warmup** (5 min)
   - Start server
   - Run 10 simple requests
   - Record stable baseline memory

2. **Continuous Multi-Turn** (30 min)
   - Single client, 100+ turns
   - Monitor for per-turn leaks
   - Check snapshot cleanup

3. **Concurrent Users** (30 min)
   - 10 concurrent clients
   - Each does 20 turns
   - Monitor for connection leaks

4. **Large Context Stress** (20 min)
   - 8k token context windows
   - Monitor snapshot size growth
   - Check tier manager cleanup

5. **Snapshot Tier Cycling** (20 min)
   - Create many snapshots
   - Force WARM → COLD transitions
   - Verify cleanup of orphaned files

---

## Commands to Set Up Monitoring

### Quick Start (Manual)

```bash
# Terminal 1: Start metrics collection
while true; do
  clear
  echo "=== $(date) ==="
  echo "--- GPU ---"
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
  echo "--- RAM ---"
  free -h | grep Mem
  echo "--- Process ---"
  ps aux | grep sglang | head -3
  sleep 5
done

# Terminal 2: Run load test
python test/phases/phase-10-scaling.py
```

### Automated (Agent-Based)

```python
# test/phases/phase-10-scaling.py
import asyncio
import subprocess
import time
from datetime import datetime

class ResourceMonitor:
    def __init__(self, log_file="phase-10-metrics.log"):
        self.log_file = log_file
        self.running = False

    async def start(self):
        self.running = True
        with open(self.log_file, "w") as f:
            f.write("timestamp,gpu_vram_mb,gpu_util_pct,ram_mb,proc_rss_mb\n")

        while self.running:
            timestamp = datetime.now().isoformat()
            gpu_vram = self._get_gpu_vram()
            gpu_util = self._get_gpu_util()
            ram_mb = self._get_ram()
            proc_rss = self._get_proc_rss()

            with open(self.log_file, "a") as f:
                f.write(f"{timestamp},{gpu_vram},{gpu_util},{ram_mb},{proc_rss}\n")

            await asyncio.sleep(5)

    def _get_gpu_vram(self):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0

    def _get_gpu_util(self):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        return int(result.stdout.strip()) if result.returncode == 0 else 0

    def _get_ram(self):
        result = subprocess.run(
            ["free", "-m"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return int(result.stdout.split('\n')[1].split()[1])
        return 0

    def _get_proc_rss(self):
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'sglang' in line and 'grep' not in line:
                    return int(line.split()[5]) // 1024  # KB to MB
        return 0

    def stop(self):
        self.running = False
```

---

## Next Steps

1. **Check for larger Mamba models** available for download
   - Search HuggingFace for "mamba" models >4B
   - Check if SGLang supports them

2. **Create phase-10 test script** with monitoring built in

3. **Run baseline memory profile** before stress testing

4. **Document findings** in `test/phases/results/phase-10-results.md`

---

## Questions to Answer

| Question | How to Test | Success Criteria |
|----------|-------------|------------------|
| Does memory leak over long runs? | 1hr+ continuous monitoring | Flat memory curve after warmup |
| Do snapshots get orphaned? | Check /tmp/mamba_snapshots after tests | Only expected snapshots exist |
| Does VRAM grow with context size? | Compare small vs XL context | Predictable growth, no leaks |
| Do concurrent requests cause leaks? | 10 parallel clients | Memory returns to baseline after |
| Does container balloon? | Monitor cgroup limits | Within expected bounds |
| Is cleanup working? | Force tier transitions | No orphaned files |

---

## Prerequisites

- [ ] Larger Mamba model downloaded (or skip)
- [ ] Agent monitoring script created
- [ ] Baseline metrics recorded
- [ ] Load test scenarios defined
