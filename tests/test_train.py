import torch

from tinylaw import train


def _args(data_dir, out_dir, *extra):
    return train.parser().parse_args([
        "--data-dir", str(data_dir), "--out-dir", str(out_dir), "--batch", "4",
        "--context", "32", "--n-layer", "2", "--n-embd", "32", "--n-head", "2",
        "--vocab-size", "400", "--warmup", "2", "--log-every", "2", "--eval-every", "4",
        "--eval-batches", "2", "--threads", "1", *extra])


def test_loss_decreases(tiny_data_dir, tmp_path):
    out = tmp_path / "ck"
    train.train(_args(tiny_data_dir, out, "--steps", "60", "--lr", "3e-3"))
    lines = (out / "train.log").read_text(encoding="utf-8").splitlines()
    losses = [float(l.split("train_loss=")[1].split()[0]) for l in lines if "train_loss=" in l]
    assert losses[-1] < losses[0] - 1.0
    assert "val_loss=" in lines[-2]
    assert lines[0][:4].isdigit() and lines[0][10] == "T"  # ISO timestamp


def test_resume_is_exact(tiny_data_dir, tmp_path):
    full = tmp_path / "full"
    train.train(_args(tiny_data_dir, full, "--steps", "12"))
    part = tmp_path / "part"
    # stop after 5 steps by saving every step and cutting total steps
    a = _args(tiny_data_dir, part, "--steps", "12", "--save-minutes", "0")
    a_stop = _args(tiny_data_dir, part, "--steps", "12", "--save-minutes", "0")
    # run 5 steps: emulate interruption by training a 12-step schedule but stopping early
    import tinylaw.train as t

    orig = t.save_checkpoint
    calls = {"n": 0}

    class Stop(Exception):
        pass

    def save_then_stop(path, state):
        orig(path, state)
        calls["n"] += 1
        if state["step"] == 5:
            raise Stop

    t.save_checkpoint = save_then_stop
    try:
        try:
            t.train(a)
        except Stop:
            pass
    finally:
        t.save_checkpoint = orig
    assert torch.load(part / "latest.pt", weights_only=False)["step"] == 5
    a_stop.resume = True
    t.train(a_stop)
    s1 = torch.load(full / "latest.pt", weights_only=False)
    s2 = torch.load(part / "latest.pt", weights_only=False)
    assert s1["step"] == s2["step"] == 12
    for k in s1["model"]:
        assert torch.allclose(s1["model"][k], s2["model"][k], atol=1e-6), k


def test_lr_schedule():
    assert train.lr_at(0, 100, 1.0, 10, 0.1) == 0.1
    assert abs(train.lr_at(9, 100, 1.0, 10, 0.1) - 1.0) < 1e-9
    assert abs(train.lr_at(100, 100, 1.0, 10, 0.1) - 0.1) < 1e-9
