from services.inference.status import RuntimeStatusRegistry


def test_runtime_status_defaults_to_onnx_backend():
    registry = RuntimeStatusRegistry()
    registry.register(
        "/private/weights/panel_label/best.onnx",
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert registry.public_snapshot() == [
        {
            "model": "best.onnx",
            "backend": "onnx",
            "providers": [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        }
    ]


def test_runtime_status_keeps_same_named_models_as_separate_entries():
    registry = RuntimeStatusRegistry()
    registry.register("/weights/a/model.onnx", ["CUDAExecutionProvider"])
    registry.register("/weights/b/model.onnx", ["CUDAExecutionProvider"])

    assert len(registry.public_snapshot()) == 2


def test_runtime_status_clear_removes_registered_models():
    registry = RuntimeStatusRegistry()
    registry.register("/weights/model.onnx", ["CPUExecutionProvider"])

    registry.clear()

    assert registry.public_snapshot() == []
