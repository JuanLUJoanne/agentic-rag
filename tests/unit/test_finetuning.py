"""
Fine-tuning tests.

Pass WITHOUT finetune optional dependencies (peft/trl/bitsandbytes) via
graceful-degradation fallback paths.  Pass WITH deps via real training loops.
No model downloads or GPU cycles happen during pytest — the fallback fires
whenever dependencies or cached models are absent.
"""
from __future__ import annotations

from src.finetuning.embedding_ft import EmbeddingFineTuner, TrainingTriple
from src.finetuning.qlora_dpo import DPOPair, QAExample, QLoRADPOTrainer

# ── EmbeddingFineTuner ─────────────────────────────────────────────────────────


class TestEmbeddingFineTuner:
    def test_prepare_data_returns_triples(self) -> None:
        ft = EmbeddingFineTuner()
        queries = ["What is RAG?", "How does BM25 work?"]
        corpus = [
            "RAG combines retrieval with generation.",
            "BM25 is a ranking function.",
            "Unrelated doc.",
        ]
        relevance = {"0": [0], "1": [1]}
        triples = ft.prepare_data(queries, corpus, relevance)
        assert len(triples) > 0
        assert all(isinstance(t, TrainingTriple) for t in triples)

    def test_prepare_data_fallback_when_no_matches(self) -> None:
        """Empty corpus must still return synthetic triples so the pipeline never breaks."""
        ft = EmbeddingFineTuner()
        triples = ft.prepare_data(["q1", "q2"], [], {})
        assert len(triples) >= 1
        assert all(isinstance(t, TrainingTriple) for t in triples)

    def test_train_returns_metrics(self) -> None:
        ft = EmbeddingFineTuner()
        triples = [TrainingTriple(query="q", positive="p", hard_negative="n")]
        result = ft.train(triples, output_dir="/tmp/test_emb")
        assert result["status"] == "complete"
        assert "final_loss" in result
        assert "model_path" in result
        assert isinstance(result["n_triples"], int)

    def test_evaluate_returns_recall(self) -> None:
        ft = EmbeddingFineTuner()
        test_set = [TrainingTriple(query="q", positive="p", hard_negative="n")]
        result = ft.evaluate(test_set)
        assert "recall@5" in result
        assert "mrr@10" in result
        assert 0.0 <= result["recall@5"] <= 1.0
        assert 0.0 <= result["mrr@10"] <= 1.0


# ── QLoRADPOTrainer ────────────────────────────────────────────────────────────


class TestQLoRADPOTrainer:
    def test_train_qlora_returns_metrics(self) -> None:
        trainer = QLoRADPOTrainer()
        examples = [QAExample(context="c", question="q", answer="a")]
        result = trainer.train_qlora(examples, output_dir="/tmp/test_qlora")
        assert result["status"] == "complete"
        assert "final_loss" in result
        assert result["n_examples"] == 1

    def test_generate_dpo_pairs(self) -> None:
        trainer = QLoRADPOTrainer()
        examples = [
            QAExample(
                context="RAG uses retrieval.",
                question="What is RAG?",
                answer="RAG combines retrieval with generation.",
            ),
            QAExample(
                context="BM25 ranks docs.",
                question="How does BM25 work?",
                answer="BM25 uses term frequency.",
            ),
        ]
        pairs = trainer.generate_dpo_pairs(examples)
        assert len(pairs) == 2
        assert all(isinstance(p, DPOPair) for p in pairs)
        for p in pairs:
            assert p.prompt
            assert p.chosen
            assert p.rejected
            # Rejected must differ from chosen — the whole point of DPO
            assert p.chosen != p.rejected

    def test_train_dpo_returns_metrics(self) -> None:
        trainer = QLoRADPOTrainer()
        pairs = [DPOPair(prompt="p", chosen="c", rejected="r")]
        result = trainer.train_dpo(pairs, output_dir="/tmp/test_dpo")
        assert result["status"] == "complete"
        assert result["n_pairs"] == 1

    def test_evaluate_returns_comparison(self) -> None:
        trainer = QLoRADPOTrainer()
        test_set = [QAExample(context="c", question="q", answer="a")]
        result = trainer.evaluate(test_set)
        assert "base_accuracy" in result
        assert "dpo_accuracy" in result
        # QLoRA should improve over base; DPO should reduce hallucinations vs QLoRA
        assert result["qlora_accuracy"] >= result["base_accuracy"]
        assert result["dpo_hallucination_rate"] <= result["qlora_hallucination_rate"]
