import numpy as np
import torch
from vdcp.b5_archive import postprocess,THRESHOLDS
from vdcp.kaggle_b5 import original_tensor
from vdcp.group_evaluation import score_final_masks

def test_channel_zero_and_native_normalization():
    image=np.zeros((256,1600,3),np.uint8);image[:,:,0]=128;image[:,:,1:]=255
    actual=original_tensor(image)
    assert actual.shape==(1,256,1600) and actual.dtype==torch.float32
    assert abs(actual[0,0,0].item()-((128/255-.3439)/.0383))<1e-5

def test_threshold_and_area_filtering():
    p=np.zeros((4,256,1600),np.float32)
    p[0,10:26,10:26]=1
    p[1,50:60,50:60]=1
    m=postprocess(p)
    assert m[0].sum()==256 and m[1].sum()==0

def test_four_supplied_evaluator_cases():
    z=np.zeros((4,256,1600),np.uint8);g=z.copy();g[0,50:70,100:180]=1
    wrong=z.copy();wrong[2,50:70,100:180]=1
    extra=g.copy();extra[0,50:70,900:980]=1
    for pred,expected in [(g,(1,1,0,1)),(wrong,(1,1,0,0)),(np.ones_like(z),(1,0,4,0)),(extra,(1,1,1,1))]:
        r=score_final_masks(g,pred)
        assert tuple(r[k] for k in ['n_gt','detected','false_alarms','correct_class'])==expected


def test_frozen_dynamic_helper_closure():
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('archive_closure_compute',root/'scripts/cloud_b5_compute.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trainer,study=module.engine.frozen_helpers(root)
    assert callable(trainer.optimizer_for) and callable(study.full_precision_class)
    assert callable(trainer.seed_everything) and callable(trainer.rng_state)


def preparation_api():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_b5_data.py'
    spec=importlib.util.spec_from_file_location('archive_prepare',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_missing_positive_annotation_is_rejected():
    import pytest
    row=dict(image_id='x.jpg',c1='1',c2='0',c3='0',c4='0')
    with pytest.raises(ValueError,match='presence differs'):preparation_api().decode_packed_for_row(row,{})

def test_annotation_presence_must_match_all_split_classes():
    import pytest
    api=preparation_api();row=dict(image_id='x.jpg',c1='0',c2='0',c3='0',c4='0')
    with pytest.raises(ValueError,match='presence differs'):api.decode_packed_for_row(row,{'x.jpg':['1 1','','','']})
    row['c1']='1';assert api.decode_packed_for_row(row,{'x.jpg':['1 1','','','']}).sum()==1
    row['c1']='0';assert api.decode_packed_for_row(row,{}).sum()==0

def test_duplicate_annotation_is_rejected(tmp_path):
    import pytest
    path=tmp_path/'train.csv';path.write_text('ImageId_ClassId,EncodedPixels\nx.jpg_1,1 1\nx.jpg_1,2 1\n')
    with pytest.raises(ValueError,match='Duplicate'):preparation_api().load_annotations(path)

def test_packaged_reference_retains_fixed_hash():
    import hashlib
    from vdcp import group_evaluation as g
    assert hashlib.sha256(g._packaged_reference.read_bytes()).hexdigest()==g.EVALUATOR_SHA256
