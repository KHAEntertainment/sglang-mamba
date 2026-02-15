# Troubleshooting Guide: Stateful Mamba Snapshots

Common issues and solutions for the snapshot system.

## Table of Contents

- [Common Issues](#common-issues)
- [Error Messages](#error-messages)
- [Memory Problems](#memory-problems)
- [Performance Issues](#performance-issues)
- [Disk Persistence Issues](#disk-persistence-issues)
- [Debugging Tools](#debugging-tools)
- [FAQ](#faq)

## Common Issues

### Issue: SnapshotDisabledError

**Symptom**: Getting `SnapshotDisabledError` when calling `save_snapshot()`.

**Cause**: Snapshots are not enabled for the engine or request.

**Solution**:

```python
# Option 1: Enable globally
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True  # Make sure this is set
)

# Option 2: Enable per-request
@function
def my_function(s):
    s.enable_snapshots()  # Enable for this request
    snap_id = s.save_snapshot()
    return snap_id
```

**Verify snapshots are enabled**:

```python
from sglang.snapshot import SNAPSHOT_AVAILABLE

if not SNAPSHOT_AVAILABLE:
    print("Snapshots not available in this SGLang version")
    print("Please upgrade to v0.5.0 or later")
else:
    print("Snapshots are available")
```

---

### Issue: SnapshotNotFoundError

**Symptom**: `SnapshotNotFoundError` when calling `restore_snapshot()`.

**Cause**: Snapshot ID doesn't exist (deleted or never created).

**Solution**:

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Check if snapshot exists before restoring
snapshot_id = "my_snapshot"
if manager.get_snapshot(snapshot_id):
    s.restore_snapshot(snapshot_id)
else:
    print(f"Snapshot {snapshot_id} not found")
    # Fallback logic
```

**Debug**:

```python
# List all available snapshots
manager = SnapshotManager(engine)
snapshots = manager.list_snapshots()
print(f"Available snapshots: {[s.snapshot_id for s in snapshots]}")
```

---

### Issue: OutOfMemoryError

**Symptom**: `OutOfMemoryError` when creating snapshots.

**Cause**: Insufficient GPU memory for snapshot storage.

**Solutions**:

#### 1. Reduce Memory Allocation

```python
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    mem_fraction_static=0.80,  # Reduce from 0.9
    enable_mamba_snapshots=True,
    snapshot_config={
        "max_snapshot_memory_gb": 3.0,  # Limit snapshot memory
    }
)
```

#### 2. Clean Up Old Snapshots

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Delete old snapshots
import time
cutoff = time.time() - 3600  # 1 hour ago
deleted = manager.delete_snapshots_before(cutoff)
print(f"Deleted {deleted} old snapshots")
```

#### 3. Enable Copy-on-Write

```python
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "enable_cow": True,  # Reduces memory duplication
    }
)
```

#### 4. Monitor Memory Usage

```python
manager = SnapshotManager(engine)

# Check memory usage
total_memory = manager.get_total_snapshot_memory()
print(f"Snapshot memory: {total_memory / 1e9:.2f} GB")

num_snapshots = len(manager.list_snapshots())
print(f"Number of snapshots: {num_snapshots}")

# Average per snapshot
if num_snapshots > 0:
    avg_memory = total_memory / num_snapshots
    print(f"Average per snapshot: {avg_memory / 1e6:.2f} MB")
```

---

### Issue: Snapshot Restore Changes Behavior

**Symptom**: After restoring a snapshot, generation produces different results than expected.

**Cause**: Misunderstanding of snapshot scope or incorrect snapshot restoration.

**Solution**:

1. **Verify snapshot metadata**:

```python
manager = SnapshotManager(engine)
snapshot = manager.get_snapshot(snapshot_id)

print(f"Snapshot ID: {snapshot.snapshot_id}")
print(f"Tokens: {len(snapshot.token_ids)}")
print(f"Sequence length: {snapshot.seq_len}")
print(f"Metadata: {snapshot.metadata}")
print(f"Age: {snapshot.age:.2f} seconds")
```

2. **Check for snapshot corruption**:

```python
try:
    s.restore_snapshot(snapshot_id)
except SnapshotInvalidError:
    print("Snapshot is corrupted")
    # Re-generate from scratch
```

3. **Ensure deterministic generation** (if needed):

```python
@function
def deterministic_generation(s, snapshot_id):
    s.restore_snapshot(snapshot_id)
    # Use temperature=0 for deterministic output
    s += gen("response", max_tokens=100, temperature=0.0)
    return s
```

---

### Issue: Slow Snapshot Operations

**Symptom**: `save_snapshot()` or `restore_snapshot()` is slow.

**Diagnosis**:

```python
import time

@function
def benchmark_snapshot(s):
    s += gen("text", max_tokens=100)

    # Benchmark save
    start = time.time()
    snap_id = s.save_snapshot()
    save_time = time.time() - start
    print(f"Save time: {save_time*1000:.2f}ms")

    # Benchmark restore
    start = time.time()
    s.restore_snapshot(snap_id)
    restore_time = time.time() - start
    print(f"Restore time: {restore_time*1000:.2f}ms")

    return s
```

**Solutions**:

1. **Disable COW if restore is slow**:

```python
# COW makes save fast but restore slower
engine = Engine(
    model_path="state-spaces/mamba-2.8b",
    enable_mamba_snapshots=True,
    snapshot_config={
        "enable_cow": False,  # Faster restore, slower save
    }
)
```

2. **Check for disk I/O bottleneck**:

```python
# Disable auto-persistence if enabled
snapshot_config={
    "auto_persist": False,  # Don't write to disk automatically
}
```

3. **Reduce snapshot frequency**:

```python
# Don't create snapshots too frequently
@function
def efficient_snapshots(s):
    for i in range(100):
        s += gen(f"text_{i}", max_tokens=10)
        # Only snapshot every 10 iterations
        if i % 10 == 0:
            s.save_snapshot(f"checkpoint_{i}")
    return s
```

---

## Error Messages

### "Snapshot ID already exists"

**Full Error**: `ValueError: Snapshot ID 'my_snapshot' already exists`

**Cause**: Trying to create a snapshot with an ID that's already in use.

**Solution**:

```python
# Option 1: Use auto-generated IDs
snap_id = s.save_snapshot()  # Auto-generates unique ID

# Option 2: Check before creating
manager = SnapshotManager(engine)
if manager.get_snapshot("my_snapshot") is None:
    snap_id = s.save_snapshot("my_snapshot")
else:
    # Use existing or generate new ID
    snap_id = s.save_snapshot()  # Auto-generate
```

---

### "Snapshot is currently being used"

**Full Error**: `SnapshotInUseError: Snapshot cannot be deleted while in use`

**Cause**: Trying to delete a snapshot that's referenced by active requests.

**Solution**:

```python
# Wait for requests to complete before deleting
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)
snapshot = manager.get_snapshot(snapshot_id)

print(f"Reference count: {snapshot.ref_count}")
# Wait until ref_count == 1 (only registry reference)

# Or force delete (not recommended)
# This will fail if ref_count > 1
try:
    manager.delete_snapshot(snapshot_id)
except SnapshotInUseError:
    print("Snapshot still in use, will delete later")
```

---

### "Failed to deserialize snapshot"

**Full Error**: `SnapshotDeserializationError: Failed to deserialize snapshot from disk`

**Cause**: Corrupted snapshot file or version incompatibility.

**Solutions**:

1. **Check file integrity**:

```bash
# Check if file exists and has content
ls -lh ./snapshots/my_snapshot.bin
```

2. **Verify format compatibility**:

```python
# Ensure you're using the same serialization format
manager = SnapshotManager(engine)

# If saved with safetensors, must load with safetensors
try:
    snap_id = manager.load_snapshot("./snapshots/my_snapshot.bin")
except SnapshotDeserializationError as e:
    print(f"Deserialization failed: {e}")
    print("File may be corrupted or saved with different format")
```

3. **Re-generate snapshot**:

```python
# If snapshot is corrupted, re-generate from scratch
# (This is why checkpointing strategies should keep backups)
```

---

## Memory Problems

### Diagnosing Memory Issues

```python
# Comprehensive memory diagnostic
from sglang.snapshot import SnapshotManager
import torch

manager = SnapshotManager(engine)

# GPU memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU allocated: {allocated:.2f} GB")
    print(f"GPU reserved: {reserved:.2f} GB")

# Snapshot memory
snapshot_memory = manager.get_total_snapshot_memory() / 1e9
num_snapshots = len(manager.list_snapshots())
print(f"Snapshot memory: {snapshot_memory:.2f} GB")
print(f"Number of snapshots: {num_snapshots}")

if num_snapshots > 0:
    avg_per_snapshot = snapshot_memory / num_snapshots
    print(f"Average per snapshot: {avg_per_snapshot * 1000:.2f} MB")

# List snapshots by size
snapshots = manager.list_snapshots(sort_by="memory_size", ascending=False)
print("\nTop 5 largest snapshots:")
for snap in snapshots[:5]:
    size_mb = snap.memory_size / 1e6
    print(f"  {snap.snapshot_id}: {size_mb:.2f} MB")
```

### Memory Optimization Strategies

#### 1. Automatic Cleanup

```python
import time
from sglang.snapshot import SnapshotManager

class AutoCleanupManager:
    def __init__(self, engine, max_memory_gb=5.0, cleanup_interval=60):
        self.manager = SnapshotManager(engine)
        self.max_memory_gb = max_memory_gb
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()

    def cleanup_if_needed(self):
        current_memory = self.manager.get_total_snapshot_memory()
        current_memory_gb = current_memory / 1e9

        if current_memory_gb > self.max_memory_gb:
            print(f"Memory limit exceeded: {current_memory_gb:.2f} GB")
            # Delete oldest snapshots
            cutoff = time.time() - self.cleanup_interval
            deleted = self.manager.delete_snapshots_before(cutoff)
            print(f"Deleted {deleted} old snapshots")

    def save_snapshot_with_cleanup(self, s, snapshot_id=None):
        # Cleanup before saving
        self.cleanup_if_needed()

        # Save snapshot
        snap_id = s.save_snapshot(snapshot_id)

        return snap_id
```

#### 2. Snapshot Pooling

```python
class SnapshotPool:
    """Reuse snapshot IDs to limit total snapshots."""

    def __init__(self, engine, pool_size=10):
        self.manager = SnapshotManager(engine)
        self.pool_size = pool_size
        self.current_index = 0
        self.snapshot_ids = [f"pool_{i}" for i in range(pool_size)]

    def save_snapshot(self, s):
        # Reuse snapshot IDs in round-robin fashion
        snap_id = self.snapshot_ids[self.current_index]

        # Delete old snapshot with this ID if exists
        if self.manager.get_snapshot(snap_id):
            self.manager.delete_snapshot(snap_id)

        # Save new snapshot
        s.save_snapshot(snap_id)

        # Move to next ID
        self.current_index = (self.current_index + 1) % self.pool_size

        return snap_id
```

---

## Performance Issues

### Profiling Snapshot Operations

```python
import time
import torch
from contextlib import contextmanager

@contextmanager
def profile_snapshot_op(operation_name):
    """Profile a snapshot operation."""
    # GPU sync for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    elapsed_ms = (end_time - start_time) * 1000
    memory_delta_mb = (end_memory - start_memory) / 1e6

    print(f"{operation_name}:")
    print(f"  Time: {elapsed_ms:.2f}ms")
    print(f"  Memory delta: {memory_delta_mb:.2f} MB")

# Usage
@function
def profiled_snapshot_ops(s):
    s += gen("text", max_tokens=100)

    with profile_snapshot_op("Save snapshot"):
        snap_id = s.save_snapshot()

    with profile_snapshot_op("Restore snapshot"):
        s.restore_snapshot(snap_id)

    with profile_snapshot_op("Delete snapshot"):
        s.delete_snapshot(snap_id)

    return s
```

### Performance Tuning

#### Optimize for Save Speed

```python
# Fast save, slower restore
snapshot_config = {
    "enable_cow": True,       # Copy-on-write for fast save
    "async_io": True,          # Async disk I/O
    "auto_persist": False,     # Don't persist automatically
}
```

#### Optimize for Restore Speed

```python
# Slower save, fast restore
snapshot_config = {
    "enable_cow": False,       # Full copy on save for fast restore
    "max_snapshots": 20,       # Fewer snapshots in memory
}
```

#### Optimize for Memory

```python
# Minimal memory usage
snapshot_config = {
    "enable_cow": True,              # Share memory when possible
    "max_snapshots": 10,             # Low limit
    "max_snapshot_memory_gb": 2.0,   # Strict limit
    "enable_persistence": True,      # Offload to disk
    "auto_persist": True,            # Auto-persist old snapshots
}
```

---

## Disk Persistence Issues

### Issue: Slow Disk I/O

**Symptom**: `persist_snapshot()` is slow or blocks inference.

**Solutions**:

1. **Enable async I/O**:

```python
snapshot_config = {
    "async_io": True,      # Don't block
    "io_threads": 4,       # Parallel I/O
}

# Use async persist
future = manager.persist_snapshot(snapshot_id, "./path.bin", async_mode=True)
# Continue working...
future.result()  # Wait when needed
```

2. **Use faster storage**:

```python
# Use SSD or RAM disk
snapshot_config = {
    "storage_path": "/dev/shm/snapshots",  # RAM disk (Linux)
}
```

3. **Enable compression**:

```python
# Smaller files, faster I/O
snapshot_config = {
    "compression_enabled": True,
    "compression_level": 6,  # Balance size vs speed
}
```

### Issue: Disk Space Full

**Symptom**: `IOError: No space left on device`

**Solutions**:

1. **Check disk usage**:

```bash
df -h ./snapshots
```

2. **Clean up old snapshots**:

```python
import os
import time

# Delete snapshots older than 24 hours
storage_path = "./snapshots"
cutoff_time = time.time() - 86400

for filename in os.listdir(storage_path):
    filepath = os.path.join(storage_path, filename)
    if os.path.getmtime(filepath) < cutoff_time:
        os.remove(filepath)
        print(f"Deleted old snapshot: {filename}")
```

3. **Set up automatic cleanup**:

```python
from sglang.snapshot import SnapshotManager

manager = SnapshotManager(engine)

# Delete old snapshots automatically
manager.set_auto_cleanup(
    max_disk_space_gb=50.0,
    cleanup_strategy="lru"  # Least recently used
)
```

---

## Debugging Tools

### Enable Debug Logging

```python
import logging

# Enable snapshot debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("sglang.snapshot")
logger.setLevel(logging.DEBUG)

# Now you'll see detailed logs
```

### Snapshot Inspector

```python
from sglang.snapshot import SnapshotManager, get_snapshot_info

manager = SnapshotManager(engine)

def inspect_snapshot(snapshot_id):
    """Comprehensive snapshot inspection."""
    snapshot = manager.get_snapshot(snapshot_id)

    if snapshot is None:
        print(f"Snapshot {snapshot_id} not found")
        return

    info = get_snapshot_info(snapshot)

    print(f"=== Snapshot: {snapshot_id} ===")
    print(f"Age: {info['age_seconds']:.2f} seconds")
    print(f"Memory: {info['memory_mb']:.2f} MB")
    print(f"Tokens: {info['num_tokens']}")
    print(f"Sequence length: {snapshot.seq_len}")
    print(f"Reference count: {snapshot.ref_count}")
    print(f"Metadata: {snapshot.metadata}")

    # Token preview
    if 'token_text' in info:
        preview = info['token_text'][:200]
        print(f"Token preview: {preview}...")

# Usage
inspect_snapshot("my_snapshot")
```

### Health Check

```python
from sglang.snapshot import SnapshotManager

def snapshot_health_check(engine):
    """Check overall snapshot system health."""
    manager = SnapshotManager(engine)

    print("=== Snapshot System Health Check ===")

    # Total snapshots
    snapshots = manager.list_snapshots()
    print(f"Total snapshots: {len(snapshots)}")

    # Memory usage
    total_memory = manager.get_total_snapshot_memory()
    print(f"Total memory: {total_memory / 1e9:.2f} GB")

    if len(snapshots) > 0:
        avg_memory = total_memory / len(snapshots)
        print(f"Average per snapshot: {avg_memory / 1e6:.2f} MB")

    # Oldest snapshot
    oldest = min(snapshots, key=lambda s: s.timestamp)
    print(f"Oldest snapshot: {oldest.snapshot_id} ({oldest.age:.2f}s old)")

    # Reference counts
    in_use = sum(1 for s in snapshots if s.ref_count > 1)
    print(f"Snapshots in use: {in_use}")

    # Warnings
    if total_memory > 10e9:  # 10 GB
        print("⚠ WARNING: High memory usage!")

    if len(snapshots) > 100:
        print("⚠ WARNING: Many snapshots! Consider cleanup.")

    if oldest.age > 86400:  # 24 hours
        print("⚠ WARNING: Very old snapshots exist!")

    print("=== Health Check Complete ===")

# Usage
snapshot_health_check(engine)
```

---

## FAQ

### Q: Can I use snapshots with transformer models?

**A**: No, snapshots are Mamba-specific. Transformer models use KV cache, which is already optimized by the radix cache system.

---

### Q: How much memory does each snapshot use?

**A**: Typically 50-100 MB for Mamba-2.8B, varies by model size and sequence length. Use `snapshot.memory_size` to check.

---

### Q: Can I share snapshots between different models?

**A**: No, snapshots are model-specific. They contain SSM states that are only compatible with the exact model architecture.

---

### Q: What happens if I restore a snapshot and then save a new one?

**A**: The new snapshot includes all state up to that point, including the restored state. Snapshots can be chained this way.

---

### Q: Are snapshots serializable for remote storage (S3, etc.)?

**A**: Yes, after saving to disk, you can upload the file to remote storage. Just ensure you load it back to the same model architecture.

```python
# Save locally
manager.persist_snapshot(snap_id, "./temp.bin")

# Upload to S3 (pseudo-code)
import boto3
s3 = boto3.client('s3')
s3.upload_file("./temp.bin", "my-bucket", "snapshots/snap.bin")

# Later, download and load
s3.download_file("my-bucket", "snapshots/snap.bin", "./downloaded.bin")
loaded_id = manager.load_snapshot("./downloaded.bin")
```

---

### Q: Can I snapshot in the middle of a generation?

**A**: You can save a snapshot at any point, but it captures state up to the last completed token. Mid-generation snapshots capture state before the current `gen()` call completes.

---

### Q: How do I debug "snapshot state mismatch" errors?

**A**: This usually means the snapshot was created with a different model or configuration. Ensure:
- Same model architecture
- Same model weights
- Same SGLang version
- Snapshot not corrupted

---

## Getting Help

If you're still experiencing issues:

1. **Check logs**: Enable debug logging to see detailed error messages
2. **GitHub Issues**: [Search or create an issue](https://github.com/sgl-project/sglang/issues)
3. **Discussions**: [Ask the community](https://github.com/sgl-project/sglang/discussions)
4. **Slack**: [Join #support channel](https://slack.sglang.io/)

When reporting issues, please include:
- SGLang version
- Model name
- Minimal reproduction code
- Full error traceback
- GPU memory usage
- Snapshot configuration

## See Also

- [User Guide](user_guide.md) - Usage patterns and best practices
- [API Reference](api_reference.md) - Detailed API documentation
- [Architecture](architecture.md) - System design and internals
- [Migration Guide](migration_guide.md) - Upgrading to snapshot support
