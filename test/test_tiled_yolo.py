from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from schemas.exceptions import ModelInferenceError
from services.inference import TensorInfo
from services.tiled_yolo import TiledYoloInfer
from services.yolo import YoloInfer


class Runner:
    input_infos = (TensorInfo('images', (1, 3, 32, 32), 'tensor(float)'),)
    output_infos = (TensorInfo('output0', (1, 6, 1), 'tensor(float)'),)
    providers = ('FakeExecutionProvider',)

    def __init__(self):
        self.calls = []

    def run(self, inputs):
        self.calls.append(inputs)
        return [np.zeros((1, 6, 1), dtype=np.float32)]

    def close(self):
        pass


def model(**kwargs):
    detector = TiledYoloInfer(2, Runner(), tiled_inference=True, tile_overlap=20, **kwargs)
    detector.id2name = {0: 'present', 1: 'no_present'}
    return detector


@pytest.fixture
def local_box(monkeypatch):
    monkeypatch.setattr('services.tiled_yolo.run_yolo_nms',
                        lambda *_args, **_kwargs: [
                            np.array([[1, 2, 5, 8, 0.8, 0]], dtype=np.float32)])
    monkeypatch.setattr('services.tiled_yolo.restore_yolo_boxes',
                        lambda boxes, *_args: boxes.copy())


def test_preprocess_creates_four_individual_inputs_and_request_metadata():
    detector = model()
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    tensors, meta = detector.preprocess(image)
    assert len(tensors) == len(meta.tiles) == 4
    assert all(tensor.shape == (1, 3, 32, 32) for tensor in tensors)
    assert all(item.src_shape == (50, 60, 3) for _, item in meta.tiles)
    assert meta.src_shape == image.shape
    assert meta.ori_img is None
    assert detector.runner.calls == []
    assert 'infer' not in TiledYoloInfer.__dict__


def test_stage_order_and_full_image_coordinates(local_box):
    events = []

    class Ordered(TiledYoloInfer):
        def preprocess(self, image):
            events.append('preprocess')
            return super().preprocess(image)

        def post_process(self, outputs, meta):
            events.append('post_process')
            return super().post_process(outputs, meta)

    runner = Runner()
    original_run = runner.run
    runner.run = lambda inputs: (events.append('run'), original_run(inputs))[1]
    detector = Ordered(2, runner, tiled_inference=True, tile_overlap=20)
    detector.id2name = {0: 'present'}
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    result = detector.infer(image)
    assert events == ['preprocess', 'run', 'run', 'run', 'run', 'post_process']
    assert result.boxes == [[1, 2, 5, 8], [1, 32, 5, 38],
                            [41, 2, 45, 8], [41, 32, 45, 38]]
    assert result.class_names == ['present'] * 4
    assert result.class_ids == [0] * 4
    assert len(result.scores) == 4
    assert result.ori_img is image


def test_small_image_uses_single_run(local_box):
    detector = model()
    detector.infer(np.zeros((10, 100, 3), dtype=np.uint8))
    assert len(detector.runner.calls) == 1


def test_middle_tile_failure_stops_and_never_postprocesses(monkeypatch):
    detector = model()
    original_run = detector.runner.run
    error = ModelInferenceError('tile failed')

    def run(inputs):
        if len(detector.runner.calls) == 1:
            raise error
        return original_run(inputs)

    detector.runner.run = run
    post = []
    monkeypatch.setattr(detector, 'post_process', lambda *_args: post.append(True))
    with pytest.raises(ModelInferenceError) as caught:
        detector.infer(np.zeros((80, 100, 3), dtype=np.uint8))
    assert caught.value is error
    assert len(detector.runner.calls) == 1
    assert not post


def test_unexpected_tile_failure_is_wrapped():
    detector = model()
    detector.runner.run = lambda *_args: (_ for _ in ()).throw(RuntimeError('backend'))
    with pytest.raises(ModelInferenceError) as caught:
        detector.infer(np.zeros((80, 100, 3), dtype=np.uint8))
    assert caught.value.context['original_error'] == 'backend'


def test_mismatched_tile_outputs_are_rejected():
    detector = model()
    _, meta = detector.preprocess(np.zeros((80, 100, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        detector.post_process([], meta)


def test_disabled_mode_matches_original_yolo():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    original = YoloInfer(2, Runner())
    detector = TiledYoloInfer(2, Runner())
    original.id2name = detector.id2name = {0: 'present', 1: 'no_present'}
    tensor, metadata = original.preprocess(image)
    other_tensor, other_meta = detector.preprocess(image)
    np.testing.assert_array_equal(tensor, other_tensor)
    assert metadata == other_meta
    result = detector.infer(image)
    assert len(detector.runner.calls) == 1
    assert result.boxes == original.infer(image).boxes == []


def test_empty_detection_results_preserve_original_image():
    detector = model()
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    result = detector.infer(image)
    assert result.boxes == result.scores == result.class_ids == result.class_names == []
    assert result.ori_img is image


def test_concurrent_requests_keep_shapes_and_offsets_local(local_box):
    barrier = Barrier(2)

    class Concurrent(TiledYoloInfer):
        def preprocess(self, image):
            prepared = super().preprocess(image)
            barrier.wait(timeout=10)
            return prepared

    detector = Concurrent(2, Runner(), tiled_inference=True, tile_overlap=20)
    detector.id2name = {0: 'present'}
    images = [np.zeros((80, 100, 3), dtype=np.uint8),
              np.zeros((120, 160, 3), dtype=np.uint8)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(detector.infer, images))
    assert first.ori_img is images[0] and second.ori_img is images[1]
    assert first.boxes[-1] == [41, 32, 45, 38]
    assert second.boxes[-1] == [71, 52, 75, 58]
    assert not any(name in detector.__dict__ for name in ('tiles', 'meta', 'src_shape', 'ori_img'))


@pytest.mark.parametrize('kwargs', [
    {'task': 'seg'}, {'tile_overlap': -1},
])
def test_invalid_tiled_detector_options(kwargs):
    with pytest.raises(ValueError):
        TiledYoloInfer(2, Runner(), tiled_inference=True, **kwargs)
