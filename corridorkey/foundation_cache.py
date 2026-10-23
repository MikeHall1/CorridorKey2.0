"""Optional job-local IID foundation reuse; RVM never enters this cache."""
import torch

from .cache import ByteBudgetCache


class FoundationWindowCache:
    def __init__(self, max_bytes, *, persistent=None):
        self.cache = ByteBudgetCache(max_bytes)
        self.persistent = persistent
        self.frames_evaluated = 0
        self.frames_attempted = 0
        self.frame_requests = 0
        self.payload_bytes_stored = 0
        self.payload_bytes_reused = 0

    @torch.no_grad()
    def window(self, display, start, *, device, channels, grid_hw, extract, retain_from=None):
        """Return full native-timeline features, extracting only cache misses.

        extract consumes an unchanged subset of source tensors and returns
        normalized float32 NCHW features. Changed extraction batch shapes can
        affect GPU numerical roundoff; this path is opt-in until validated.
        """
        count = display.shape[0]
        self.frame_requests += count
        if self.cache.max_bytes == 0 and self.persistent is None:
            self.frames_attempted += count
            result = extract(display)
            self.frames_evaluated += count
            return result
        outputs = {name: torch.empty((count, width, *grid_hw), device=device, dtype=torch.float32)
                   for name, width in channels.items() if width > 0}
        missing = []
        for local in range(count):
            payload = self.cache.get(start + local)
            durable_hit = False
            if payload is None and self.persistent is not None:
                payload = self.persistent.get(start + local, display[local], channels, grid_hw)
                durable_hit = payload is not None
            if payload is None:
                missing.append(local)
            else:
                for name in outputs:
                    # Blocking copy completes before the borrowed CPU payload
                    # can be evicted. No retained CPU references escape.
                    outputs[name][local].copy_(payload[name])
                    self.payload_bytes_reused += payload[name].numel() * payload[name].element_size()
                if durable_hit and (retain_from is None or start + local >= retain_from):
                    size = sum(value.numel() * value.element_size() for value in payload.values())
                    self.cache.store(start + local, size, lambda payload=payload: payload)
                del payload
        if retain_from is not None:
            # Hits have already been copied into independent working tensors.
            self.cache.discard_before(retain_from)
        if missing:
            selected = display if len(missing) == count else display[missing]
            self.frames_attempted += len(missing)
            features = extract(selected)
            self.frames_evaluated += len(missing)
            if set(features) != set(outputs):
                raise ValueError("foundation output names do not match configured feature channels")
            for name, value in features.items():
                expected = (len(missing), channels[name], *grid_hw)
                if value.dtype != torch.float32 or tuple(value.shape) != expected:
                    raise ValueError(f"{name}: expected normalized float32 features {expected}, got {value.dtype} {tuple(value.shape)}")
            for offset, local in enumerate(missing):
                for name in outputs:
                    outputs[name][local].copy_(features[name][offset])
                if self.persistent is not None:
                    self.persistent.put(start + local, display[local], channels, grid_hw,
                                        {name: value[offset] for name, value in features.items()},
                                        producer_frames=[start + item for item in missing])
                if retain_from is not None and start + local < retain_from:
                    continue
                size = sum(value[offset].numel() * value.element_size() for value in features.values())

                def independent_cpu_frame(offset=offset):
                    return {name: value[offset].detach().to(device="cpu", copy=True).contiguous()
                            for name, value in features.items()}

                if self.cache.store(start + local, size, independent_cpu_frame):
                    self.payload_bytes_stored += size
        return outputs

    def snapshot(self):
        stats = self.cache.snapshot()
        # Disabled mode uses the exact original extraction batch and no lookups.
        if self.cache.max_bytes == 0:
            stats.update({"misses": self.frame_requests, "hit_rate": 0.0 if self.frame_requests else None})
        return {**stats, "mode": ("durable_and_ram_float32" if self.cache.max_bytes else "durable_float32")
                if self.persistent is not None else "ram_float32" if self.cache.max_bytes else "off",
                "scope": ("bounded job-local RAM plus sealed IID C-RADIO/MoGe frames; no recurrent resume"
                          if self.persistent is not None else
                          "one job, IID C-RADIO/MoGe only; no playback or cross-job reuse"),
                "frame_requests": self.frame_requests, "foundation_frames_evaluated": self.frames_evaluated,
                "foundation_frames_attempted": self.frames_attempted,
                "payload_bytes_stored": self.payload_bytes_stored, "payload_bytes_reused": self.payload_bytes_reused,
                "budget_scope": "retained CPU tensor payload; excludes working buffers and Python overhead"}

    def close(self):
        self.cache.clear()
