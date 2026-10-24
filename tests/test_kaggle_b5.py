import importlib
import numpy as np
import pytest
import torch
from pathlib import Path
import random


def module():
    return importlib.import_module('vdcp.kaggle_b5')


def test_original_single_channel_and_normalization():
    bgr = np.array([[[100, 150, 200]]], dtype=np.uint8)
    tensor = module().original_tensor(bgr)
    assert tensor.shape == (1, 1, 1)
    assert tensor.item() == pytest.approx((100 / 255 - .3439) / .0383, rel=1e-6)


def test_original_loss_uses_global_dice_including_empty_classes():
    logits = torch.zeros(1, 4, 2, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 0, 0] = 1
    loss = module().original_loss(logits, target)
    expected_dice = 1 - 1 / 9
    assert loss.item() == pytest.approx(.6 * np.log(2) + .4 * expected_dice)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_empty_mask_still_has_finite_supervision():
    logits = torch.zeros(2, 4, 2, 2, requires_grad=True)
    loss = module().original_loss(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert loss.item() == pytest.approx(.6 * np.log(2) + .4)


def test_resume_rejects_changed_config_and_data():
    identity = {'config': 'a', 'splits': {'train': 'b'}, 'source': 'c'}
    module().verify_resume_identity({'identity': identity}, identity)
    with pytest.raises(ValueError, match='identity'):
        module().verify_resume_identity({'identity': identity}, {**identity, 'config': 'changed'})


def test_session_budget_reserves_validation_and_saving():
    assert module().can_start_epoch(remaining=1200, epoch_seconds=600, validation_seconds=300)
    assert not module().can_start_epoch(remaining=950, epoch_seconds=600, validation_seconds=300)


def test_manifest_rejects_overlap_and_path_traversal():
    module().check_records([{'image_id': 'a.jpg'}], [{'image_id': 'b.jpg'}])
    with pytest.raises(ValueError):
        module().check_records([{'image_id': 'a.jpg'}], [{'image_id': 'a.jpg'}])
    with pytest.raises(ValueError):
        module().check_records([{'image_id': '../a.jpg'}], [{'image_id': 'b.jpg'}])


def test_historical_fpn_single_channel_cat_shape_and_roundtrip():
    torch.set_num_threads(2)
    model = module().HistoricalFPN().eval()
    assert model.encoder._conv_stem.in_channels == 1
    assert model.decoder.merge_policy == 'cat'
    assert model.decoder.final_conv.in_channels == 512
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 64, 96))
    assert output.shape == (1, 4, 64, 96)
    assert torch.isfinite(output).all()
    clone = module().HistoricalFPN().eval()
    clone.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(output, clone(torch.zeros(1, 1, 64, 96)))


def trainer():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/train_kaggle_b5.py'
    spec=importlib.util.spec_from_file_location('kaggle_trainer',path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_full_checkpoint_restores_optimizer_scheduler_rng(tmp_path):
    script=trainer()
    model=torch.nn.Linear(3,1)
    optimizer=torch.optim.RAdam(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e: 1/(e+1))
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    model(torch.ones(2,3)).sum().backward()
    optimizer.step(); scheduler.step()
    identity={'config':'a','source':'b','splits':{'train':'c','val':'d'}}
    script.save_full(tmp_path/'last.pt',model,optimizer,scheduler,scaler,7,identity,18,{'dice':.8})
    expected=(random.random(),np.random.rand(),torch.rand(2))
    expected_weights={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.zero_()
    optimizer.param_groups[0]['lr']=100
    payload=script.restore(tmp_path/'last.pt',model,optimizer,scheduler,scaler,identity)
    actual=(random.random(),np.random.rand(),torch.rand(2))
    assert actual[:2]==expected[:2] and torch.equal(actual[2],expected[2])
    assert optimizer.param_groups[0]['lr']==.005
    assert scheduler.last_epoch==1 and payload['epoch']==7
    assert optimizer.state
    assert all(torch.equal(v,expected_weights[k]) for k,v in model.state_dict().items())


def test_schedule_has_original_flat_stage_and_logistic_tail():
    from scipy.stats import logistic
    config={'flat_epochs':30,'anneal_epochs':221}
    fn=trainer().lr_factor
    assert fn(29,config)==1
    for epoch in (30,31,140,200,250):
        assert fn(epoch,config)==pytest.approx(logistic.sf(epoch-30,221*.5,221*.1))


def test_radam_weight_decay_does_not_contaminate_moments():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=torch.nn.Linear(1,1,bias=False)
            self.decoder=torch.nn.Linear(1,1,bias=False)
    model=Tiny()
    config={'encoder_lr':.1,'decoder_lr':.2,'encoder_weight_decay':.3,'decoder_weight_decay':.4}
    optimizer=trainer().optimizer_for(model,config)
    with torch.no_grad():
        model.encoder.weight.fill_(1.)
        model.decoder.weight.fill_(1.)
    for p in model.parameters():
        p.grad=torch.zeros_like(p)
    optimizer.step()
    assert model.encoder.weight.item()==pytest.approx(.97)
    assert model.decoder.weight.item()==pytest.approx(.92)
    assert all(torch.count_nonzero(s['exp_avg'])==0 and torch.count_nonzero(s['exp_avg_sq'])==0
               for s in optimizer.state.values())


def test_resume_identity_binds_actual_masks_and_selection_protocol(tmp_path):
    for directory in ('src','scripts','data/splits','data/raw/train_images','data/raw/masks','reference'):
        (tmp_path/directory).mkdir(parents=True,exist_ok=True)
    (tmp_path/'src/dummy.py').write_text('a=1')
    (tmp_path/'scripts/train_kaggle_b5.py').write_text('a=2')
    for split,name in (('train','a'),('val','b'),('test','c')):
        (tmp_path/f'data/splits/{split}.csv').write_text(f'image_id\n{name}.jpg\n')
    for name in ('a','b'):
        (tmp_path/f'data/raw/train_images/{name}.jpg').write_bytes(b'image')
        (tmp_path/f'data/raw/masks/{name}.png').write_bytes(b'mask')
    (tmp_path/'weights.pt').write_bytes(b'weights')
    protocol=tmp_path/'reference/locked_protocol.json'
    protocol.write_text('{}')
    config={'pretrained_file':'weights.pt','validation_reference_protocol':'reference/locked_protocol.json'}
    first=module().experiment_identity(tmp_path,config)
    (tmp_path/'data/raw/masks/a.png').write_bytes(b'changed mask')
    second=module().experiment_identity(tmp_path,config)
    assert first['selected_data']!=second['selected_data']
    protocol.write_text('{"threshold":0.5}')
    third=module().experiment_identity(tmp_path,config)
    assert second['reference_protocol']!=third['reference_protocol']


def test_notebook_ignores_old_extracted_source_when_resuming(tmp_path):
    import json
    input_dir=tmp_path/'input'
    for root in (input_dir/'private/vdcp_b5',input_dir/'previous/input_extracted/vdcp_b5'):
        (root/'configs').mkdir(parents=True)
        (root/'configs/fpn_v9_kaggle_original.json').write_text('{}')
        (root/'bundle_manifest.json').write_text(json.dumps(
            {'train_images':8797,'validation_images':1885,'test_images_included':0,'files':{}}))
    notebook=Path(__file__).resolve().parents[1]/'kaggle/b5_fpn_t4.ipynb'
    code=''.join(json.loads(notebook.read_text(encoding='utf-8'))['cells'][2]['source'])
    code=code.replace('/kaggle/input',input_dir.as_posix()).replace('/kaggle/working',tmp_path.as_posix())
    namespace={'Path':Path,'json':json}
    exec(compile(code,'bundle-discovery','exec'),namespace)
    assert namespace['ROOT']==input_dir/'private/vdcp_b5'


def test_packaged_notebook_has_valid_python_and_no_auto_extension():
    import ast,json
    notebook=Path(__file__).resolve().parents[1]/'kaggle/b5_fpn_t4.ipynb'
    data=json.loads(notebook.read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in data['cells'] if c['cell_type']=='code')
    ast.parse(code)
    assert 'STOP_AT=60' in code and 'SESSION_HOURS=10.5' in code
    assert "'--resume'" in code and 'sha256' in code
