import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from vdcp.kaggle_b5 import original_loss


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/cloud_b5_compute.py'
    assert path.is_file(), 'The separate cloud computation module is required'
    spec = importlib.util.spec_from_file_location('cloud_compute_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Toy(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(1, 5, 1),
            torch.nn.BatchNorm2d(5), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Conv2d(5, 4, 1))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.layers(x)


def samples(count):
    x = torch.randn(count, 1, 3, 5)
    y = torch.zeros(count, 4, 3, 5)
    y[0, 0, :1] = 1
    y[-1, 2, :, :4] = 1
    return x, y


def scaler():
    return torch.amp.GradScaler('cuda', enabled=False)


@pytest.mark.parametrize('count', [18, 13, 1])
@pytest.mark.parametrize('micro', [18, 9, 6, 3])
def test_loss_gradients_and_radam_update_match_original_whole_group(count, micro):
    m = driver()
    torch.manual_seed(41)
    a = Toy(); b = copy.deepcopy(a)
    x, y = samples(count)
    m.engine.freeze_batchnorm(b)
    oa = torch.optim.RAdam(a.parameters(), lr=.02, decoupled_weight_decay=True)
    ob = torch.optim.RAdam(b.parameters(), lr=.02, decoupled_weight_decay=True)
    expected = original_loss(b(x), y)
    expected.backward(); ob.step()
    before = m.engine.bn_snapshot(a)
    got = m.update_group(a, oa, scaler(), x, y, micro, 'cpu')
    assert isinstance(got, float)
    assert got == pytest.approx(expected.item(), abs=2e-7)
    for actual, reference in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, atol=3e-8, rtol=2e-5)
        torch.testing.assert_close(actual, reference, atol=3e-8, rtol=2e-5)
    m.engine.assert_bn_unchanged(a, before)
    assert a.layers[1].weight.requires_grad and a.layers[1].weight.grad is not None
    assert a.calls == (1 if micro >= count else 2 * ((count + micro - 1) // micro))


@pytest.mark.parametrize('micro', [18, 9, 6, 3])
def test_dropout_replay_preserves_gradients_and_advances_rng_once(micro):
    m = driver()
    torch.manual_seed(81)
    a = Toy(.4); b = copy.deepcopy(a)
    x, y = samples(13)
    m.engine.freeze_batchnorm(b)
    initial = m.engine.capture_rng('cpu')
    logits = torch.cat([b(xx) for xx in x.split(micro)])
    after = m.engine.capture_rng('cpu')
    expected = original_loss(logits, y)
    expected.backward()
    ob = torch.optim.SGD(b.parameters(), lr=.02); ob.step()
    m.engine.restore_rng(initial, 'cpu')
    oa = torch.optim.SGD(a.parameters(), lr=.02)
    got = m.update_group(a, oa, scaler(), x, y, micro, 'cpu')
    assert got == pytest.approx(expected.item(), abs=2e-7)
    assert m.engine.states_equal(m.engine.capture_rng('cpu'), after)
    for actual, reference in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, atol=3e-8, rtol=2e-5)
        torch.testing.assert_close(actual, reference, atol=3e-8, rtol=2e-5)


def test_saved_optimizer_scaler_and_rng_reproduce_next_stochastic_update(tmp_path):
    m = driver(); torch.manual_seed(17)
    a = Toy(.3); oa = torch.optim.RAdam(a.parameters(), lr=.02)
    sa = scaler(); x, y = samples(13)
    m.update_group(a, oa, sa, x, y, 6, 'cpu')
    path = tmp_path / 'state.pt'
    torch.save(dict(model=a.state_dict(), optimizer=oa.state_dict(), scaler=sa.state_dict(),
                    rng=m.engine.capture_rng('cpu')), path)
    expected = m.update_group(a, oa, sa, x, y, 6, 'cpu')
    expected_rng = m.engine.capture_rng('cpu')
    saved = torch.load(path, weights_only=False)
    b = Toy(.3); ob = torch.optim.RAdam(b.parameters(), lr=.9); sb = scaler()
    b.load_state_dict(saved['model']); ob.load_state_dict(saved['optimizer'])
    sb.load_state_dict(saved['scaler']); m.engine.restore_rng(saved['rng'], 'cpu')
    assert m.update_group(b, ob, sb, x, y, 6, 'cpu') == expected
    assert m.engine.states_equal(a.state_dict(), b.state_dict())
    assert m.engine.states_equal(oa.state_dict(), ob.state_dict())
    assert m.engine.states_equal(sa.state_dict(), sb.state_dict())
    assert m.engine.states_equal(expected_rng, m.engine.capture_rng('cpu'))


def test_autocast_is_disabled_and_parent_bn_buffers_remain_exact():
    m = driver(); a = Toy(); opt = torch.optim.SGD(a.parameters(), lr=.02)
    x, y = samples(13); before = m.engine.bn_snapshot(a)
    observed = []
    a.register_forward_hook(lambda _, args, result: observed.append(result.dtype))
    for micro in (18, 6):
        a.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            m.update_group(a, opt, scaler(), x, y, micro, 'cpu')
        m.engine.assert_bn_unchanged(a, before)
    assert set(observed) == {torch.float32}
    assert not a.layers[1].training and a.layers[1].bias.requires_grad


def test_missing_optimizer_parameters_and_nonfp32_inputs_are_rejected():
    m = driver(); a = Toy(); x, y = samples(13)
    incomplete = torch.optim.SGD(a.layers[-1].parameters(), lr=.02)
    with pytest.raises(ValueError, match='optimizer|coverage'):
        m.update_group(a, incomplete, scaler(), x, y, 18, 'cpu')
    opt = torch.optim.SGD(a.parameters(), lr=.02)
    for bad_x, bad_y in ((x.half(), y), (x, y.half())):
        with pytest.raises(ValueError, match='FP32'):
            m.update_group(a, opt, scaler(), bad_x, bad_y, 18, 'cpu')


@pytest.mark.parametrize('count,micro', [(0, 18), (19, 18), (13, 7)])
def test_invalid_group_or_physical_batch_is_rejected(count, micro):
    m = driver(); a = Toy(); opt = torch.optim.SGD(a.parameters(), lr=.02)
    with pytest.raises(ValueError):
        m.update_group(a, opt, scaler(), torch.zeros(count, 1, 3, 5),
                       torch.zeros(count, 4, 3, 5), micro, 'cpu')


@pytest.mark.parametrize('micro', [18, 6])
def test_nonfinite_logits_do_not_step_optimizer(micro):
    m = driver(); a = Toy(); x, y = samples(13)
    a.layers[-1].bias.data.fill_(float('inf'))
    opt = torch.optim.SGD(a.parameters(), lr=.02)
    before = a.layers[0].weight.detach().clone()
    with pytest.raises(FloatingPointError):
        m.update_group(a, opt, scaler(), x, y, micro, 'cpu')
    assert torch.equal(a.layers[0].weight, before)


def test_changed_replay_predictions_are_rejected_before_optimizer_step():
    m = driver()
    class Drifting(Toy):
        def forward(self, x):
            result = super().forward(x)
            return result + (1 if torch.is_grad_enabled() else 0)
    a = Drifting(); opt = torch.optim.SGD(a.parameters(), lr=.02)
    x, y = samples(13); before = a.layers[0].weight.detach().clone()
    with pytest.raises(RuntimeError, match='replay'):
        m.update_group(a, opt, scaler(), x, y, 6, 'cpu')
    assert torch.equal(a.layers[0].weight, before)


def test_nonfinite_python_scalar_optimizer_state_is_rejected():
    m = driver(); a = Toy(); opt = torch.optim.SGD(a.parameters(), lr=.02)
    opt.state[next(a.parameters())]['diagnostic_scalar'] = float('nan')
    x, y = samples(13)
    with pytest.raises(FloatingPointError, match='state'):
        m.update_group(a, opt, scaler(), x, y, 18, 'cpu')
