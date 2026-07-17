from unittest.mock import MagicMock, patch

from services.inference import OnnxRuntimeOptions, RunnerSpec


def test_dc_fuse_builds_and_injects_an_explicit_onnx_runner():
    from services.dc_fuse.business_logic import DCFuseDetectorAPI

    settings = MagicMock()
    runner = object()
    with (
        patch(
            "services.dc_fuse.business_logic.create_inference_runner",
            return_value=runner,
        ) as factory,
        patch("services.dc_fuse.business_logic.DCFuseDetector") as detector,
    ):
        DCFuseDetectorAPI(settings)

    options = OnnxRuntimeOptions.from_settings(settings)
    factory.assert_called_once_with(
        RunnerSpec(
            scenario="dc_fuse",
            onnx_path="./weights/dc_fuse/det_yolo_v5.onnx",
        ),
        options,
    )
    detector.assert_called_once_with(
        0.6,
        runner=runner,
    )


def test_dc_fuse_closes_runner_when_detector_initialization_fails():
    from schemas.exceptions import ModelInferenceError
    from services.dc_fuse.business_logic import DCFuseDetectorAPI

    settings = MagicMock()
    runner = MagicMock()
    with (
        patch(
            "services.dc_fuse.business_logic.create_inference_runner",
            return_value=runner,
        ),
        patch(
            "services.dc_fuse.business_logic.DCFuseDetector",
            side_effect=RuntimeError("detector failed"),
        ),
    ):
        try:
            DCFuseDetectorAPI(settings)
        except ModelInferenceError:
            pass

    runner.close.assert_called_once_with()
