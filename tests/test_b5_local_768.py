import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/continue_b5_local_768.py'
    assert path.exists(), 'Corrected single-768 adapter is missing'
    spec = importlib.util.spec_from_file_location('corrected_768', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(dropout=0):
    return torch.nn.Sequential(torch.nn.Conv2d(1, 5, 1), torch.nn.BatchNorm2d(5),
        torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Conv2d(5, 4, 1))


@pytest.mark.parametrize('count,micro,foreground', [(18,1,True),(5,2,True),(7,3,False),(1,1,True)])
def test_global_dice_true_full_group_gradient_and_update(count, micro, foreground):
    m = driver(); torch.manual_seed(4)
    a = model(); b = copy.deepcopy(a)
    x = torch.randn(count,1,3,5); y = torch.zeros(count,4,3,5)
    if foreground:
        y[0,0,:1] = 1
        if count > 1: y[-1,2,:,:4] = 1
    m.freeze_batchnorm(a); m.freeze_batchnorm(b)
    oa = torch.optim.RAdam(a.parameters(), lr=.1, decoupled_weight_decay=True)
    ob = torch.optim.RAdam(b.parameters(), lr=.1, decoupled_weight_decay=True)
    from vdcp.kaggle_b5 import original_loss
    expected = original_loss(b(x),y); expected.backward(); ob.step()
    buffers = m.bn_snapshot(a)
    got = m.update_group(a,oa,torch.amp.GradScaler('cuda',enabled=False),x,y,micro,'cpu')
    assert got == pytest.approx(expected.item(), abs=2e-7)
    for aa,bb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(aa.grad,bb.grad,atol=3e-8,rtol=2e-5)
        torch.testing.assert_close(aa,bb,atol=3e-8,rtol=2e-5)
    m.assert_bn_unchanged(a,buffers)
    assert a[1].weight.grad is not None


def test_dropout_replays_same_predictions_and_advances_rng_once():
    m=driver(); torch.manual_seed(81)
    a=model(.4); b=copy.deepcopy(a)
    x=torch.randn(5,1,4,3); y=torch.zeros(5,4,4,3); y[-1,0]=1
    m.freeze_batchnorm(a); m.freeze_batchnorm(b)
    state=torch.get_rng_state()
    logits=torch.cat([b(xx) for xx in x.split(2)])
    after=torch.get_rng_state()
    from vdcp.kaggle_b5 import original_loss
    loss=original_loss(logits,y); loss.backward()
    ob=torch.optim.SGD(b.parameters(),lr=.02); ob.step()
    torch.set_rng_state(state)
    oa=torch.optim.SGD(a.parameters(),lr=.02)
    got=m.update_group(a,oa,torch.amp.GradScaler('cuda',enabled=False),x,y,2,'cpu')
    assert got==pytest.approx(loss.item(),abs=2e-7)
    assert torch.equal(torch.get_rng_state(),after)
    for aa,bb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(aa,bb,atol=2e-8,rtol=2e-6)
    # update_group must reapply eval to BN after every caller train().
    before=m.bn_snapshot(a); a.train()
    m.update_group(a,oa,torch.amp.GradScaler('cuda',enabled=False),x,y,1,'cpu')
    m.assert_bn_unchanged(a,before)


def test_config_and_identity_bind_all_execution_inputs():
    m=driver(); cfg=m.load_config(m.ROOT/'configs/local_b5_768_bnfix.json')
    base={'source':m.PARENT_SOURCE}; runtime={'device':'test'}
    identity=m.identity_for(base,runtime,'adapter','config','gate',cfg)
    assert identity['crop_width']==768 and identity['effective_batch_size']==18
    assert identity['parent_checkpoint_sha256']==m.PARENT_SHA
    for key,value in [('precision','amp'),('crop_width',416),('effective_batch_size',16)]:
        bad={**cfg,key:value}
        with pytest.raises(ValueError):m.validate_config(bad)
    with pytest.raises(ValueError):m.identity_for({'source':'bad'},runtime,'a','c','g',cfg)


def test_single_device_rng_ignores_cloud_device_one(monkeypatch):
    m=driver(); states=[torch.tensor([3],dtype=torch.uint8),torch.tensor([7],dtype=torch.uint8)]
    calls=[]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda state,device:calls.append((state,device)))
    state=m.capture_rng('cpu'); state['cuda']=states
    m.restore_rng(state,'cuda:0')
    assert len(calls)==1 and torch.equal(calls[0][0],states[0]) and calls[0][1]==0
    with pytest.raises(ValueError):m.restore_rng({**state,'cuda':[]},'cuda:0')


def test_checkpoint_exact_restore_pending_and_identity_rejection(tmp_path):
    m=driver(); torch.manual_seed(71)
    a=model(.3); oa=torch.optim.RAdam(a.parameters(),lr=.02,decoupled_weight_decay=True)
    sa=torch.optim.lr_scheduler.StepLR(oa,1,.9); scaler=torch.amp.GradScaler('cuda',enabled=False)
    x=torch.randn(3,1,4,4); y=torch.zeros(3,4,4,4); y[0,1]=1
    m.update_group(a,oa,scaler,x,y,1,'cpu'); sa.step()
    identity={'test':'exact'}; row={'epoch':73,'validation':None}
    m.save_state(tmp_path/'last.pt',a,oa,sa,scaler,73,identity,{'region_f2':.6},{},row,pending=True,device='cpu')
    expected_rng=m.capture_rng('cpu')
    m.update_group(a,oa,scaler,x,y,1,'cpu')
    b=model(.3); ob=torch.optim.RAdam(b.parameters(),lr=.9,decoupled_weight_decay=True)
    sb=torch.optim.lr_scheduler.StepLR(ob,1,.9)
    p=m.load_state(tmp_path/'last.pt',b,ob,sb,scaler,identity,'cpu')
    assert p['pending_validation'] and sb.state_dict()==sa.state_dict()
    assert torch.equal(torch.get_rng_state(),expected_rng['torch'])
    m.update_group(b,ob,scaler,x,y,1,'cpu')
    assert m.states_equal(a.state_dict(),b.state_dict())
    assert m.states_equal(oa.state_dict(),ob.state_dict())
    with pytest.raises(ValueError):m.load_state(tmp_path/'last.pt',b,ob,sb,scaler,{'bad':1},'cpu')
    assert m.next_action({'epoch':73,'pending_validation':True},73)=='validate'
    assert m.next_action({'epoch':73,'pending_validation':False},73)=='complete'


def test_baseline_gate_requires_canonical_complete_bound_full_validation(tmp_path):
    m=driver(); metrics=dict(images=1885,n_gt=3029,region_f2=.59,recall=.58,false_alarm_rate=.35,
        kaggle_style_mean_dice_at_05=.7)
    gate=dict(status='complete',parent_sha256=m.PARENT_SHA,canonical_precision='fp32_direct',
        reference_protocol_sha256=m.PROTOCOL_SHA,fp32_direct=metrics)
    p=tmp_path/'gate.json'; p.write_text(json.dumps(gate))
    assert m.read_baseline_gate(p)==metrics
    for field,value in [('status','running'),('parent_sha256','bad'),('canonical_precision','fp16_cache'),('reference_protocol_sha256','bad')]:
        p.write_text(json.dumps({**gate,field:value}))
        with pytest.raises(ValueError):m.read_baseline_gate(p)


def test_quality_gate_and_explicit_long_continuation():
    m=driver(); cfg=m.load_config(m.ROOT/'configs/local_b5_768_bnfix.json')
    base=dict(region_f2=.6,recall=.6,false_alarm_rate=.3)
    assert not m.regression(base,base,cfg)
    for key,value in [('region_f2',.589),('recall',.579),('false_alarm_rate',.401)]:
        assert m.regression({**base,key:value},base,cfg)
    assert m.arguments([]).target_epoch==73
    with pytest.raises(SystemExit):m.arguments(['--target-epoch','80'])
    with pytest.raises(SystemExit):m.arguments(['--resume','--probe-only'])
    assert m.arguments(['--resume','--target-epoch','80']).target_epoch==80


def test_fp32_postprocessing_does_not_quantize_boundary():
    m=driver(); p=np.zeros((4,2,2),np.float32);p[0,0,0]=np.float32(.74999)
    assert m.fp32_postprocess(p,[.75]*4,[1]*4).sum()==0


def test_resume_rejects_saved_config_or_scheduler_phase_mismatch(tmp_path):
    m=driver(); a=model(); opt=torch.optim.RAdam(a.parameters(),lr=.1)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,1)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    import hashlib
    cfg={'original':True}
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    identity={'schema':'local_b5_768_bnfix_v1','base_experiment':{'config':config_hash}}
    path=tmp_path/'last.pt'
    m.save_state(path,a,opt,scheduler,scaler,73,identity,{'region_f2':.5},{'tampered':True},{'epoch':73},device='cpu')
    with pytest.raises(ValueError,match='config|phase'):m.load_state(path,a,opt,scheduler,scaler,identity,'cpu')
    m.save_state(path,a,opt,scheduler,scaler,73,identity,{'region_f2':.5},cfg,{'epoch':73},device='cpu')
    with pytest.raises(ValueError,match='phase'):m.load_state(path,a,opt,scheduler,scaler,identity,'cpu')


def test_pending_validation_recovery_preserves_rng_and_publishes_best(tmp_path):
    m=driver(); a=model(); opt=torch.optim.RAdam(a.parameters(),lr=.1)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,1);scaler=torch.amp.GradScaler('cuda',enabled=False)
    identity={'test':'pending'}; row={'epoch':73,'validation':None}
    m.save_state(tmp_path/'last.pt',a,opt,scheduler,scaler,73,identity,{'region_f2':.5,'dice':.7},{},row,pending=True,device='cpu')
    p=torch.load(tmp_path/'last.pt',weights_only=False)
    before=m.capture_rng('cpu')
    metrics=dict(images=1885,n_gt=3029,region_f2=.6,kaggle_style_mean_dice_at_05=.7,
        canonical_precision='fp32_direct',reference_protocol_sha256=m.PROTOCOL_SHA)
    def validator():
        torch.rand(3);np.random.rand();return metrics
    result=m.finish_validation(tmp_path,p,a,opt,scheduler,scaler,validator,(),'cpu')
    assert not result['pending_validation']
    assert m.states_equal(before,m.capture_rng('cpu'))
    best=torch.load(tmp_path/'best_region_f2.pt',weights_only=False)
    assert best['best']['region_f2']==.6 and not best['pending_validation']
    assert json.loads((tmp_path/'epochs.jsonl').read_text())['epoch']==73
    # Completed checkpoints must reconstruct an interrupted/missing history write.
    (tmp_path/'epochs.jsonl').unlink()
    m.reconcile_history(tmp_path,result)
    assert json.loads((tmp_path/'epochs.jsonl').read_text())['epoch']==73


def test_parent_migration_uses_new_baseline_instead_of_saved_cloud_metrics():
    m=driver(); baseline={'region_f2':.61,'kaggle_style_mean_dice_at_05':.8}
    parent={'best':{'region_f2':.9},'last_epoch_record':{'validation':{'region_f2':.9}},'identity':{'cloud':1}}
    p=m.migrate_parent(parent,{'new':1},baseline)
    assert p['best']['region_f2']==.61 and p['last_epoch_record']['validation']==baseline


def test_probe_output_cannot_target_production_tree():
    m=driver()
    with pytest.raises(SystemExit):m.arguments(['--probe-only','--probe-output',str(m.ROOT/'outputs/other_production')])


def test_baseline_binding_rejects_different_data_or_precision_environment():
    m=driver();base={'data':'same'}
    environment=dict(python='p',torch='t',cuda='c',numpy='n',opencv='o',scipy='s',cudnn=9,gpu='g',
        tf32_matmul=False,tf32_cudnn=True,cudnn_benchmark=False,cudnn_deterministic=False)
    runtime={**environment,'matmul_allow_tf32':False,'cudnn_allow_tf32':True}
    gate=dict(base_experiment=base,environment=environment,test_accessed=False)
    m.verify_baseline_binding(gate,base,runtime)
    with pytest.raises(ValueError):m.verify_baseline_binding(gate,{'data':'changed'},runtime)
    with pytest.raises(ValueError):m.verify_baseline_binding(gate,base,{**runtime,'cudnn_allow_tf32':False})




