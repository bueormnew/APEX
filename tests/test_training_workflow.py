import torch
from torch.utils.data import DataLoader

from apex_core import APEXConfig, APEXModel, APEXTrainer, TrainingConfig
from apex_core.data import ByteTokenizer, RandomTextWindowDataset
from hybrid_model import HybridCausalLM, HybridLMConfig


def test_byte_tokenizer_round_trip_and_shifted_windows(tmp_path):
    tokenizer = ByteTokenizer()
    text = "APEX: café 🧠"
    assert tokenizer.decode(tokenizer.encode(text)) == text

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    assert ByteTokenizer.load(path).decode(tokenizer.encode(text)) == text

    dataset = RandomTextWindowDataset(list(range(20)), sequence_length=5, samples=4, random_sampling=False)
    inputs, labels = dataset[0]
    assert inputs.shape == labels.shape == (5,)
    assert torch.equal(inputs[1:], labels[:-1])


def test_apex_full_sequence_matches_cached_generation_and_is_causal():
    torch.manual_seed(7)
    config = HybridLMConfig(
        vocab_size=41,
        d_model=16,
        max_seq_len=24,
        layer_pattern=["lrcm", "hopmix", "mamba3"],
        echo_n_keys=8,
        echo_top_k=2,
        echo_rank=4,
        hop_gate_heads=4,
        lrcm_heads=4,
        lrcm_local_window=4,
        lrcm_chunk_size=4,
        lrcm_desc_dim=8,
        lrcm_beam=2,
        mamba_d_state=8,
        mamba_headdim=8,
    )
    model = HybridCausalLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 18))

    with torch.no_grad():
        logits = model(tokens)["logits"]
        states = model.init_inference_states(2, tokens.device)
        incremental = []
        for index in range(tokens.shape[1]):
            step_logits, states = model.step(tokens[:, index:index + 1], states)
            incremental.append(step_logits)
        torch.testing.assert_close(torch.cat(incremental, dim=1), logits, atol=1e-5, rtol=1e-5)

        altered = tokens.clone()
        altered[:, 12] = (altered[:, 12] + 1) % config.vocab_size
        altered_logits = model(altered)["logits"]
        torch.testing.assert_close(altered_logits[:, :12], logits[:, :12], atol=1e-5, rtol=1e-5)


def test_trainer_saves_resumable_checkpoint_and_inference_model(tmp_path):
    torch.manual_seed(11)
    config = APEXConfig(
        vocab_size=32,
        d_model=16,
        max_seq_len=8,
        preset=None,
        layer_pattern=["mamba3"],
        echo_n_keys=8,
        echo_top_k=2,
        echo_rank=4,
        mamba_d_state=8,
        mamba_headdim=8,
        mamba_is_mimo=False,
    )
    model = APEXModel(config)
    tokens = torch.randint(0, config.vocab_size, (64,))
    data = DataLoader(
        RandomTextWindowDataset(tokens, sequence_length=8, samples=6),
        batch_size=2,
    )
    checkpoint = tmp_path / "training_state.pt"
    trainer = APEXTrainer(
        model,
        TrainingConfig(
            learning_rate=1e-3,
            warmup_steps=0,
            total_steps=1,
            epochs=1,
            device="cpu",
            training_checkpoint_path=str(checkpoint),
            checkpoint_every=1,
            log_interval=1,
        ),
    )
    history = trainer.train(data)
    assert len(history["step"]) == 1
    saved_model = model.save_apex(str(tmp_path / "nested" / "model.apex"))
    reloaded = APEXModel.load_apex(saved_model)
    assert reloaded.config.to_dict() == config.to_dict()
    sample = torch.randint(0, config.vocab_size, (1, 6))
    with torch.no_grad():
        torch.testing.assert_close(model(sample)["logits"], reloaded(sample)["logits"])

    resumed = APEXTrainer(
        APEXModel(config),
        TrainingConfig(
            learning_rate=1e-3,
            warmup_steps=0,
            total_steps=3,
            epochs=1,
            device="cpu",
            resume_checkpoint_path=str(checkpoint),
        ),
    )
    resumed_history = resumed.train(data)
    assert resumed_history["step"][-1] == 3
    assert resumed_history["learning_rate"][1] <= 1.1e-5
