"""Regression for finite FP32 weights whose FP16 inference overflows."""
import numpy as np
import pytest
import torch
from vdcp.kaggle_b5 import original_probability


class OverflowNet(torch.nn.Module):
    def __init__(self,value):
        super().__init__()
        self.conv=torch.nn.Conv2d(1,4,1,bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(value)

    def forward(self,images):
        result=self.conv(images)
        return result-result


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires genuine CUDA FP16 overflow')
def test_fp16_overflow_retries_finite_fp32_inference():
    model=OverflowNet(70000.).cuda().eval()
    image=np.full((8,8,3),100,dtype=np.uint8)
    result=original_probability(model,image,'cuda')
    assert result.shape==(4,8,8)
    assert np.array_equal(result,np.full((4,8,8),.5,dtype=np.float32))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires CUDA')
def test_bad_fp32_weights_still_fail_instead_of_fabricating_masks():
    model=OverflowNet(float('nan')).cuda().eval()
    with pytest.raises(FloatingPointError,match='Nonfinite'):
        original_probability(model,np.full((8,8,3),100,dtype=np.uint8),'cuda')


def test_finite_cpu_inference_does_not_retry():
    class CountingNet(OverflowNet):
        calls = 0
        def forward(self, images):
            self.calls += 1
            return super().forward(images)
    model = CountingNet(1.).eval()
    retries = []
    probability = original_probability(model, np.full((8,8,3),100,dtype=np.uint8),
                                       'cpu', on_fp32_retry=lambda: retries.append(True))
    assert model.calls == 1 and retries == [] and np.isfinite(probability).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA FP16 overflow')
def test_full_validation_records_retry_image_and_cache_policy(tmp_path):
    import cv2
    import importlib.util
    import json
    from pathlib import Path
    for directory in ('data/raw/train_images', 'data/raw/masks', 'reference'):
        (tmp_path/directory).mkdir(parents=True)
    image = np.full((8,8,3),100,dtype=np.uint8)
    mask = np.zeros((8,8), dtype=np.uint8)
    mask[2:4,2:4] = 1
    assert cv2.imwrite(str(tmp_path/'data/raw/train_images/a.jpg'), image)
    assert cv2.imwrite(str(tmp_path/'data/raw/masks/a.png'), mask)
    (tmp_path/'data/splits').mkdir()
    (tmp_path/'data/splits/val.csv').write_text('image_id\na.jpg\n')
    (tmp_path/'reference/protocol.json').write_text(json.dumps({'thresholds':[.6]*4,'min_areas':[1]*4}))
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_bytes(b'checkpoint-fixture')
    path = Path(__file__).resolve().parents[1]/'scripts/train_kaggle_b5.py'
    spec = importlib.util.spec_from_file_location('precision_validation', path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    cache = tmp_path/'cache'
    result = trainer.validate(OverflowNet(70000.).cuda(), tmp_path, [{'image_id':'a.jpg'}],
        {'validation_reference_protocol':'reference/protocol.json', 'input_convention':'original'},
        'cuda', cache, checkpoint)
    assert result['images'] == 1 and result['fp32_retry_count'] == 1
    assert result['fp32_retry_images'] == ['a.jpg']
    metadata = json.loads((cache/'metadata.json').read_text())
    assert metadata['inference']['nonfinite_policy'] == 'retry_fp32_then_raise'
    assert metadata['fp32_retry_images'] == ['a.jpg']
    assert np.isfinite(np.load(cache/'a.npz')['probabilities']).all()
    with pytest.raises(FloatingPointError, match='a.jpg'):
        trainer.validate(OverflowNet(float('nan')).cuda(), tmp_path, [{'image_id':'a.jpg'}],
            {'validation_reference_protocol':'reference/protocol.json'}, 'cuda')
