from types import SimpleNamespace

from hip_orb.profile_head import summarize_trace


def test_profile_ranks_operators_without_double_counting_annotation_ranges():
    def event(name, device, gpu):
        return SimpleNamespace(key=name, device_type=device, count=3, input_shapes=[],
                               self_cpu_time_total=1000, self_device_time_total=gpu,
                               device_time_total=gpu, self_device_memory_usage=0)

    events = [event("hip_backward_and_optimizer", "DeviceType.CPU", 50000),
              event("head_pair_input_projection", "DeviceType.CPU", 40000),
              event("Optimizer.step#AdamW.step", "DeviceType.CPU", 30000),
              event("cuda_annotation", "DeviceType.CUDA", 20000),
              event("aten::mm", "DeviceType.CPU", 10000)]
    trace = SimpleNamespace(key_averages=lambda **kwargs: events)
    result = summarize_trace(trace)
    assert [item["name"] for item in result["top_gpu_operations"]] == ["aten::mm"]
    assert result["top_gpu_operations"][0]["gpu_self_ms"] == 10
    assert [item["name"] for item in result["scopes"]] == [
        "hip_backward_and_optimizer", "head_pair_input_projection"]
