import torch

from tinylaw.model import GPT, GPTConfig


def test_full_size_model_has_about_ten_million_parameters():
    n = GPT(GPTConfig()).num_params()
    assert 9_500_000 < n < 10_500_000


def test_forward_shapes_and_loss(tiny_model: GPT, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (2, 16))
    logits, loss = tiny_model(idx, idx)
    assert logits.shape == (2, 16, tiny_config.vocab_size)
    assert loss.item() > 0


def test_causal(tiny_model: GPT, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size, (1, 12))
    a, _ = tiny_model(idx)
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % tiny_config.vocab_size
    b, _ = tiny_model(idx2)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6)


def test_tied_embeddings(tiny_model: GPT):
    names = [n for n, _ in tiny_model.named_parameters()]
    assert not any("lm_head" in n for n in names)


def test_generate_stops_at_context(tiny_model: GPT):
    out = tiny_model.generate(torch.zeros((1, 1), dtype=torch.long), 40, top_k=5)
    assert out.shape[1] == 41
