import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/continue_b5_local_768_long.py'
    assert path.exists(), 'Dedicated 50-epoch continuation is missing'
    spec = importlib.util.spec_from_file_location('long50_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def setup_state(m, epoch=100):
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 1), torch.nn.BatchNorm2d(4))
    opt = torch.optim.RAdam(model.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1/(1+n/100))
    scheduler.last_epoch = epoch
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    cfg = {'original': True}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    identity = {'schema': m.FIXED['schema'], 'base_experiment': {'config': cfg_hash}}
    return model, opt, scheduler, scaler, cfg, identity


def test_long_plan_accepts_fifty_epochs_and_rejects_legacy_outputs():
    m = driver()
    args = m.arguments([])
    assert args.target_epoch == 122 and args.session_hours == 40
    assert m.FIXED['start_epoch'] == 72 and m.FIXED['new_epochs'] == 50
    assert m.FIXED['train_encoder'] and m.FIXED['train_decoder']
    assert m.FIXED['quality_stop'] is False
    with pytest.raises(SystemExit): m.arguments(['--target-epoch', '80'])
    with pytest.raises(SystemExit): m.arguments(['--output', str(m.ROOT/'outputs/b5_local_768_bnfix_v1')])
    with pytest.raises(SystemExit): m.arguments(['--resume', '--probe-only'])
    assert m.arguments(['--resume']).resume


def test_normal_regression_records_warning_but_never_stops():
    m = driver()
    baseline = dict(region_f2=.59, recall=.58, false_alarm_rate=.35)
    bad = dict(region_f2=.10, recall=.12, false_alarm_rate=2.0)
    decision = m.quality_observation(bad, baseline)
    assert decision['regressed'] and decision['stop'] is False
    with pytest.raises(ValueError):
        m.quality_observation({**bad, 'region_f2':float('nan')}, baseline)


def test_complete_epoch100_restore_preserves_state_and_rejects_bad_phase(tmp_path):
    m = driver()
    model,opt,scheduler,scaler,cfg,identity = setup_state(m)
    path = tmp_path/'last.pt'
    m.engine.save_state(path,model,opt,scheduler,scaler,100,identity,{'region_f2':.59},cfg,
                        {'epoch':100,'validation':{'region_f2':.57}},device='cpu')
    saved = torch.load(path, weights_only=False)
    model[0].weight.data.zero_()
    m.load_long_state(path,model,opt,scheduler,scaler,identity,'cpu')
    assert m.engine.states_equal(model.state_dict(), saved['model'])
    assert m.engine.states_equal(opt.state_dict(), saved['optimizer'])
    assert scheduler.state_dict() == saved['scheduler']
    for epoch,phase,config in [(123,123,cfg),(100,99,cfg),(100,100,{'tampered':True})]:
        altered={**saved,'epoch':epoch,'scheduler':{**saved['scheduler'],'last_epoch':phase},'config':config}
        m.engine.atomic_torch(path,altered)
        with pytest.raises(ValueError):m.load_long_state(path,model,opt,scheduler,scaler,identity,'cpu')


def test_pending_low_score_validation_preserves_best_and_snapshots(tmp_path):
    m = driver()
    model,opt,scheduler,scaler,cfg,identity=setup_state(m,73)
    good=dict(images=1885,n_gt=3029,region_f2=.59,recall=.58,false_alarm_rate=.35,
              classification_accuracy=.62,kaggle_style_mean_dice_at_05=.94,
              canonical_precision='fp32_direct',reference_protocol_sha256=m.engine.PROTOCOL_SHA)
    m.engine.save_state(tmp_path/'best_region_f2.pt',model,opt,scheduler,scaler,72,identity,
                        {'region_f2':.59,'dice':.94},cfg,{'epoch':72,'validation':good},device='cpu')
    m.engine.save_state(tmp_path/'last.pt',model,opt,scheduler,scaler,73,identity,
                        {'region_f2':.59,'dice':.94},cfg,{'epoch':73,'validation':None},pending=True,device='cpu')
    p=torch.load(tmp_path/'last.pt',weights_only=False)
    bad={**good,'region_f2':.10,'recall':.12,'false_alarm_rate':2.}
    before=m.engine.capture_rng('cpu')
    def validate():
        torch.rand(5)
        return bad
    completed=m.finish_long_validation(tmp_path,p,model,opt,scheduler,scaler,validate,(),'cpu',good)
    assert not completed['pending_validation']
    assert completed['last_epoch_record']['quality_observation']['stop'] is False
    assert m.engine.states_equal(before,m.engine.capture_rng('cpu'))
    assert torch.load(tmp_path/'best_region_f2.pt',weights_only=False)['epoch']==72
    assert torch.load(tmp_path/'last.pt',weights_only=False)['epoch']==73
    assert (tmp_path/'checkpoints/epoch073.pt').is_file()


def test_missing_epoch_history_cannot_be_skipped(tmp_path):
    m=driver()
    for epoch in [72,73,75]:
        m.engine.history_row(tmp_path,{'epoch':epoch,'validation':{'images':1885,'n_gt':3029}})
    with pytest.raises(ValueError):m.verify_history(tmp_path,75)


def test_engine_digest_and_new_policy_are_bound():
    m=driver()
    assert m.engine.sha(m.ROOT/'scripts/continue_b5_local_768.py')==m.ENGINE_SHA
    policy=m.load_policy(m.ROOT/'configs/local_b5_768_long50.json')
    assert policy==m.FIXED
    with pytest.raises(ValueError):m.validate_policy({**policy,'crop_width':416})
    with pytest.raises(ValueError):m.validate_policy({**policy,'train_encoder':False})


@pytest.mark.skip(reason="Historical epoch72 checkpoint is an external asset; not shipped in ordinary Git")
def test_real_baseline_parent_migration_and_optimizer_progress_checks():
    m=driver()
    baseline=m.read_long_baseline(m.ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    assert baseline['canonical_precision']=='fp32_direct'
    assert baseline['region_f2']==pytest.approx(.5912252541466024)
    parent=torch.load(m.ROOT/'artifacts/cloud_recovery/paired_epoch72/b5_paired_long/control/best_region_f2.pt',map_location='cpu',weights_only=False)
    identity=dict(schema=m.FIXED['schema'],policy=m.FIXED,
                  base_experiment=parent['identity']['base_experiment'])
    migrated=m.engine.migrate_parent(parent,identity,baseline)
    assert m._check_payload(migrated,identity)['epoch']==72
    states=migrated['optimizer']['state']
    key=next(iter(states)); original_step=states[key]['step'].clone()
    states[key]['step']+=1
    with pytest.raises(ValueError,match='step'):m._check_payload(migrated,identity)
    states[key]['step']=original_step
    group=migrated['optimizer']['param_groups'][0];original_lr=group['lr']
    group['lr']=.001
    with pytest.raises(ValueError,match='learning rate'):m._check_payload(migrated,identity)
    group['lr']=original_lr
    assert m._check_payload(migrated,identity)['scheduler']['last_epoch']==72


def test_complete_last_recovers_missing_history_and_snapshot(tmp_path):
    m=driver()
    model,opt,scheduler,scaler,cfg,identity=setup_state(m,73)
    baseline=m.read_long_baseline(m.ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    best={'region_f2':baseline['region_f2'],'dice':baseline['kaggle_style_mean_dice_at_05']}
    m.engine.save_state(tmp_path/'best_region_f2.pt',model,opt,scheduler,scaler,72,identity,best,cfg,
                        {'epoch':72,'validation':baseline},device='cpu')
    low={**baseline,'region_f2':.1,'recall':.1}
    m.engine.save_state(tmp_path/'last.pt',model,opt,scheduler,scaler,73,identity,best,cfg,
                        {'epoch':73,'validation':low},device='cpu')
    p=torch.load(tmp_path/'last.pt',weights_only=False)
    completed=m._publish_completed(tmp_path,p,baseline)
    assert (tmp_path/'checkpoints/epoch073.pt').is_file()
    rows=[json.loads(line) for line in (tmp_path/'epochs.jsonl').read_text().splitlines()]
    assert rows[0]['epoch']==73 and completed['last_epoch_record']['quality_observation']['stop'] is False
    m._publish_completed(tmp_path,completed,baseline)
    rows=[json.loads(line) for line in (tmp_path/'epochs.jsonl').read_text().splitlines()]
    assert len(rows)==1
    assert torch.load(tmp_path/'best_region_f2.pt',weights_only=False)['epoch']==72


def test_interrupted_initial_publication_resumes_exact_parent_state(tmp_path):
    m=driver()
    assert hasattr(m,'initialize_long_run'), 'Interrupted initialization must be recoverable'
    model,opt,scheduler,scaler,cfg,identity=setup_state(m,72)
    baseline=m.read_long_baseline(m.ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    path=tmp_path/'last.pt'
    m.engine.save_state(path,model,opt,scheduler,scaler,72,identity,
                        {'region_f2':baseline['region_f2'],'dice':baseline['kaggle_style_mean_dice_at_05']},
                        cfg,{'epoch':72,'validation':baseline},device='cpu')
    original=torch.load(path,weights_only=False)
    migrated=m.engine.migrate_parent(original,identity,baseline)
    m.engine.atomic_torch(path,migrated)
    m.initialize_long_run(tmp_path,original,identity,baseline,resume=True)
    for name in ['run_identity.json','baseline_validation.json','effective_policy.json','best_region_f2.pt']:
        assert (tmp_path/name).is_file()
    assert json.loads((tmp_path/'run_identity.json').read_text())==identity
    assert m.engine.states_equal(torch.load(path,weights_only=False),migrated)
    assert m.engine.states_equal(torch.load(tmp_path/'best_region_f2.pt',weights_only=False),migrated)


def test_interrupted_initial_publication_rejects_wrong_saved_state(tmp_path):
    m=driver()
    assert hasattr(m,'initialize_long_run'), 'Interrupted initialization must be recoverable'
    model,opt,scheduler,scaler,cfg,identity=setup_state(m,72)
    baseline=m.read_long_baseline(m.ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    path=tmp_path/'last.pt'
    m.engine.save_state(path,model,opt,scheduler,scaler,72,identity,
                        {'region_f2':baseline['region_f2'],'dice':baseline['kaggle_style_mean_dice_at_05']},
                        cfg,{'epoch':72,'validation':baseline},device='cpu')
    original=torch.load(path,weights_only=False)
    migrated=m.engine.migrate_parent(original,identity,baseline)
    migrated['model']=copy.deepcopy(migrated['model'])
    migrated['model']['0.weight'].add_(1)
    m.engine.atomic_torch(path,migrated)
    with pytest.raises(ValueError):m.initialize_long_run(tmp_path,original,identity,baseline,resume=True)
    assert not (tmp_path/'run_identity.json').exists()


@pytest.mark.parametrize('numeric_type',[np.float64,np.float32])
def test_real_validator_numpy_dice_is_valid_and_normalized(numeric_type):
    m=driver()
    metrics=m.read_long_baseline(m.ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    metrics['kaggle_style_mean_dice_at_05']=numeric_type(metrics['kaggle_style_mean_dice_at_05'])
    m._metrics(metrics,complete=True)
    normalized=m.normalize_validation_metrics(metrics)
    assert type(normalized['kaggle_style_mean_dice_at_05']) is float
    assert normalized['kaggle_style_mean_dice_at_05']==float(metrics['kaggle_style_mean_dice_at_05'])
    assert normalized['tp']==metrics['tp'] and normalized['by_class']==metrics['by_class']
    assert type(metrics['kaggle_style_mean_dice_at_05']) is numeric_type


@pytest.mark.parametrize('bad',[np.float64('nan'),np.float32('inf'),True,'0.94'])
def test_validator_rejects_nonfinite_or_non_numeric_dice(bad):
    m=driver()
    metrics=m.read_long_baseline(m.ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    metrics['kaggle_style_mean_dice_at_05']=bad
    with pytest.raises(ValueError):m._metrics(metrics,complete=True)


def test_numpy_dice_crosses_pending_validation_publication(tmp_path):
    m=driver()
    model,opt,scheduler,scaler,cfg,identity=setup_state(m,73)
    baseline=m.read_long_baseline(m.ROOT/'reports/v9/local_768_bnfix/baseline_precision.json')
    best={'region_f2':baseline['region_f2'],'dice':baseline['kaggle_style_mean_dice_at_05']}
    for name,epoch,pending in [('best_region_f2.pt',72,False),('last.pt',73,True)]:
        m.engine.save_state(tmp_path/name,model,opt,scheduler,scaler,epoch,identity,best,cfg,
                            {'epoch':epoch,'validation':None if pending else baseline},pending=pending,device='cpu')
    actual={**baseline,'kaggle_style_mean_dice_at_05':np.float64(.941234)}
    p=torch.load(tmp_path/'last.pt',weights_only=False)
    m.finish_long_validation(tmp_path,p,model,opt,scheduler,scaler,lambda:actual,(),'cpu',baseline)
    saved=torch.load(tmp_path/'last.pt',weights_only=False)
    assert not saved['pending_validation'] and saved['epoch']==73
    assert type(saved['last_epoch_record']['validation']['kaggle_style_mean_dice_at_05']) is float
    assert (tmp_path/'checkpoints/epoch073.pt').exists()
