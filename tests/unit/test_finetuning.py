"""Unit tests for fine-tuning stubs: EmbeddingFineTuner and QLoRADPOTrainer."""
from __future__ import annotations

from src.finetuning.embedding_ft import EmbeddingFineTuner, TrainingTriple
from src.finetuning.qlora_dpo import DPOPair, QAExample, QLoRADPOTrainer

# ── EmbeddingFineTuner ─────────────────────────────────────────────────────────


def test_embedding_ft_prepare_data_returns_triples() -> None:
    ft = EmbeddingFineTuner()
    queries = ["What is RAG?", "How does BM25 work?"]
    corpus = [
        "RAG retrieves documents before generating answers.",
        "BM25 uses term frequency and inverse document frequency.",
        "Unrelated paragraph about cooking.",
    ]
    relevance = {"0": [0], "1": [1]}

    triples = ft.prepare_data(queries, corpus, relevance)

    assert len(triples) >= 1
    for t in triples:
        assert isinstance(t, TrainingTriple)
        assert t.query
        assert t.positive
        assert t.hard_negative


def test_embedding_ft_prepare_data_fallback_when_no_matches() -> None:
    """When relevance is empty, prepare_data must still return mock triples."""
    ft = EmbeddingFineTuner()
    triples = ft.prepare_data(["q1", "q2"], [], {})
    assert len(triples) >= 1


def test_embedding_ft_train_returns_expected_structure() -> None:
    ft = EmbeddingFineTuner()
    triples = [
        TrainingTriple("q1", "positive 1", "negative 1"),
        TrainingTriple("q2", "positive 2", "negative 2"),
    ]
    result = ft.train(triples, output_dir="/tmp/emb_model")

    assert result["status"] == "complete"
    assert "model_path" in result
    assert "final_loss" in result
    assert isinstance(result["n_triples"], int)


def test_embedding_ft_evaluate_returns_recall_and_mrr() -> None:
    ft = EmbeddingFineTuner()
    test_set = [TrainingTriple("q", "p", "n")]
    metrics = ft.evaluate(test_set)

    assert "recall@5" in metrics
    assert "mrr@10" in metrics
    assert 0.0 <= metrics["recall@5"] <= 1.0
    assert 0.0 <= metrics["mrr@10"] <= 1.0


# ── QLoRADPOTrainer ────────────────────────────────────────────────────────────


def test_qlora_train_returns_expected_structure() -> None:
    trainer = QLoRADPOTrainer()
    examples = [
        QAExample("context A", "What is A?", "A is the first letter."),
        QAExample("context B", "What is B?", "B is the second letter."),
    ]
    result = trainer.train_qlora(examples, output_dir="/tmp/qlora")

    assert result["status"] == "complete"
    assert "model_path" in result
    assert "final_loss" in result
    assert result["n_examples"] == 2


def test_dpo_pairs_generation_returns_pairs() -> None:
    trainer = QLoRADPOTrainer()
    qa_examples = [
        QAExample("ctx", "q1", "answer 1"),
        QAExample("ctx", "q2", "answer 2"),
    ]
    pairs = trainer.generate_dpo_pairs(qa_examples)

    assert len(pairs) == 2
    for pair in pairs:
        assert isinstance(pair, DPOPair)
        assert pair.prompt
        assert pair.chosen
        assert pair.rejected
        assert pair.chosen != pair.rejected


def test_dpo_train_returns_expected_structure() -> None:
    trainer = QLoRADPOTrainer()
    pairs = [DPOPair("prompt", "chosen answer", "rejected answer")]
    result = trainer.train_dpo(pairs, qlora_model="/tmp/qlora", output_dir="/tmp/dpo")

    assert result["status"] == "complete"
    assert "model_path" in result
    assert result["n_pairs"] == 1


def test_qlora_dpo_evaluate_returns_comparison() -> None:
    """Evaluation must return metrics for base, qlora, and dpo models."""
    trainer = QLoRADPOTrainer()
    test_set = [QAExample("context", "question", "answer")]
    metrics = trainer.evaluate(test_set)

    assert "base_accuracy" in metrics
    assert "qlora_accuracy" in metrics
    assert "dpo_accuracy" in metrics
    # QLoRA should improve over base
    assert metrics["qlora_accuracy"] >= metrics["base_accuracy"]
    # DPO should reduce hallucinations vs QLoRA
    assert metrics["dpo_hallucination_rate"] <= metrics["qlora_hallucination_rate"]
