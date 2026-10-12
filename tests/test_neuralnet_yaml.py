"""Tests for owl.algorithms.neuralnet -- the YAML-configured deep-learning algos.

Encodes the repair contract from the 2026-10-11 session:
the module must work by leveraging algos configured through the YAML
config system. A YAML `model:` string like
`owl.algorithms.neuralnet.perceptronNN` must resolve to a constructible,
trainable class.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
import yaml

from owl.functions import _instantiate_from_path

ALGOS_YAML = Path(__file__).parent.parent / "owl" / "_data_" / "algos.yaml"
NN_MODULE = "owl.algorithms.neuralnet"


def _iter_owl_entries():
    """Iterate (algoN, seqN, model_path) triples for owl-platform entries."""
    with open(ALGOS_YAML) as f:
        data = yaml.safe_load(f) or {}
    triples = []
    for algoN, algo in data.items():
        if not isinstance(algo, dict) or algoN == "<(META)>":
            continue
        for seqN, seq in algo.items():
            if not isinstance(seq, dict):
                continue
            owl_plat = (seq.get("platform") or {}).get("owl")
            if not isinstance(owl_plat, dict):
                continue
            model = owl_plat.get("model")
            if isinstance(model, str) and model:
                triples.append((algoN, seqN, model))
    return triples


class TestYamlModelPathsResolve:
    """Every owl-platform `model:` string in algos.yaml must be importable."""

    def test_algos_yaml_exists(self) -> None:
        assert ALGOS_YAML.exists(), f"{ALGOS_YAML} missing"

    def test_owl_entries_exist_in_yaml(self) -> None:
        entries = _iter_owl_entries()
        assert len(entries) >= 32, (
            f"expected >= 32 owl-platform entries, got {len(entries)}"
        )

    def test_every_owl_model_path_resolves(self) -> None:
        """The core YAML-config contract -- resolves each model: to a class."""
        failures = []
        for algoN, seqN, model_path in _iter_owl_entries():
            try:
                _instantiate_from_path(model_path, {})
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    f"{algoN}/{seqN} -> {model_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
        assert not failures, (
            f"{len(failures)} of {len(_iter_owl_entries())} owl model paths "
            "failed:\n" + "\n".join(failures)
        )

    def test_no_stale_molecules_nn_path_in_yaml(self) -> None:
        """The dead `owl.molecules.logicizer.NN` prefix must be gone."""
        with open(ALGOS_YAML) as f:
            raw = f.read()
        assert "owl.molecules.logicizer.NN" not in raw, (
            "algos.yaml still references the removed owl.molecules.logicizer.NN package"
        )


class TestPerceptronNN:
    """Direct construction contract for the base NN class."""

    def test_construct_default(self) -> None:
        from owl.algorithms.neuralnet import perceptronNN

        net = perceptronNN(3, 1)
        assert net.n_FTRs == 3
        net._setActivation(None)  # must not raise

    def test_construct_with_cfg_override(self) -> None:
        from owl.algorithms.neuralnet import perceptronNN

        net = perceptronNN(3, 1, cfg={"ActiveNetwork": None})
        assert net.config is not None

    def test_actv_classes_available(self) -> None:
        """Activation classes re-exported through the package surface."""
        import owl.algorithms
        from owl.algorithms.actv import leakyrelu, linear, relu, sigmoid, swish, tanh
        for cls in (linear, sigmoid, tanh, relu, leakyrelu, swish):
            assert hasattr(cls, "fx")
            assert hasattr(cls, "dx")

    def test_package_exports(self) -> None:
        import owl.algorithms as a
        for name in (
            "perceptronNN", "RandomAlgo", "ZeroClassAlgo",
            "ZeroRegressionAlgo", "ZeroTimeSeriesAlgo",
        ):
            assert hasattr(a, name), f"owl.algorithms missing {name}"


class TestNNSubclasses:
    """Each NN subclass must be constructible (inherits perceptronNN.__init__)."""

    @pytest.mark.parametrize(
        "cls_name",
        [
            "autoencoderNN", "boltzmannmachineNN", "convolutionalNN",
            "deconvolutionalNN", "restrictedBMNN", "deepbeliefNN",
            "deepconvolutionalNN", "deepresidualNN", "denoisingAENN",
            "echostateNN", "extremelearningmachineNN",
            "generativeadversarialNN", "hopfieldNN", "kohonenNN",
            "liquidstatemachineNN", "recurrentNN", "longshortmemoryNN",
            "markovchainNN", "neuralturingmachineNN", "radialbasisNN",
            "sparseAENN", "supportvectormachineNN", "variationAENN",
        ],
    )
    def test_subclass_importable(self, cls_name: str) -> None:
        mod = import_module(NN_MODULE)
        assert hasattr(mod, cls_name), f"missing {cls_name} in {NN_MODULE}"
        cls = getattr(mod, cls_name)
        assert isinstance(cls, type)
