# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
"""
---
<(META)>:
    docid: 51e8dc2d-df40-4375-a1c6-56a34d81e9e0
    name: Owl Core
    description: >
        Machine learning orchestration engine.
        Loads algorithm configurations from YAML and trains/predicts using
        sklearn, tsfresh, and other platforms.
    version: 0.1.0
    authority: filesystem
    security: seclvl2
    <(WT)>: -32
"""
# -*- coding: utf-8 -*
# ======================================Standard Library Modules======================================================||
from __future__ import annotations

import datetime as dt
import gc
import logging
import uuid
from os.path import join
from typing import Any

from pandas import DataFrame
from kahndor.logma import Logma

from owl._platforms import load_platforms
from owl.functions import predict as _predict
from owl.functions import train as _train

# ====================================================================================================================||
logger = logging.getLogger(__name__)

here = join(__file__, '..', '')
logma = Logma(__name__)
pxcfg = join(here, '_data_', 'owl.yaml')
algos = join(here, '_data_', 'algos.yaml')
nsmbls = join(here, '_data_', 'nsmbls.yaml')
actvs = join(here, '_data_', 'activations.yaml')


class Owl:
    """Machine learning orchestration engine.
    
    Loads algorithm configurations from YAML and trains/predicts using
    sklearn, tsfresh, and other platforms.
    """
    
    def __init__(self, tag: str, cfg: dict[str, Any] | None = None) -> None:
        """Initialize Owl with a unique tag.
        
        Args:
            tag: Unique identifier for this Owl instance
            cfg: Optional configuration overrides
        """
        logma.info("Owl initialized: tag=%s", tag)
        self.tag = tag
        self.config = cfg or {}
        self.code: str | None = None
        self.cycle = str(uuid.uuid4())[:8]
        self.FTRs: DataFrame | None = None
        self.model: Any = None
        self.start = dt.datetime.now()
        self.workername = f'WRKR_{self.tag}'
        self.active_model: dict[str, str] | None = None
        self.data_library: str | None = None
        self.models: list[list[str]] = []
        self.algon: str = ''
        self.plat: str = ''
        self.platforms = load_platforms()
        logger.info("Owl initialized with tag: %s", tag)

    # --------------------------------------------------------------------------------------------------- #
    def setActiveModel(self, algoN: str, seqN: str, platN: str) -> Owl:
        """Set the active algorithm/sequence/platform combination.
        
        Args:
            algoN: Algorithm name (e.g., 'algo12Z1')
            seqN: Sequence name (e.g., 'a000')
            platN: Platform name (e.g., 'sklearn')
        """
        self.active_model = {'algoN': algoN, 'seqN': seqN, 'platN': platN}
        return self

    def setDataLibrary(self, data_library: str) -> Owl:
        """Set the data library identifier."""
        self.data_library = data_library
        return self

    def setModels(self, models: list[list[str]]) -> Owl:
        """Set the list of models to train/predict with."""
        self.models = models
        return self

    def addModel(self, algoN: str, seqN: str, platform: str) -> Owl:
        """Add a model to the models list."""
        self.models.append([algoN, seqN, platform])
        return self

    def cross_validate(self, ftrs: DataFrame, targs: DataFrame,
                       n_splits: int = 5,
                       models: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Run k-fold cross-validation across configured models.
        
        Args:
            ftrs: Features DataFrame
            targs: Targets DataFrame  
            n_splits: Number of CV folds (default: 5)
            models: List of model configs [{algoN, seqN, platN}, ...]
                    Defaults to self.models or single active model
        
        Returns:
            Dict mapping model codes to per-fold scores and aggregates
        """
        try:
            from sklearn.model_selection import KFold
        except ImportError:
            logger.error("scikit-learn required for cross_validate")
            return {}
        
        if models is None:
            if self.models:
                models = [{'algoN': m[0], 'seqN': m[1], 'platN': m[2]} for m in self.models]
            elif self.active_model:
                models = [self.active_model]
            else:
                logger.error("No models configured for cross-validation")
                return {}
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results: dict[str, Any] = {}
        
        for model_cfg in models:
            algoN = model_cfg['algoN']
            seqN = model_cfg.get('seqN', 'a000')
            platN = model_cfg.get('platN', 'sklearn')
            
            self.setActiveModel(algoN, seqN, platN)
            code = f'{algoN}_{seqN}_{platN}'
            logger.info("CV: %s with %s folds", code, n_splits)
            
            fold_scores: list[dict[str, float]] = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(ftrs)):
                f_train = ftrs.iloc[train_idx]
                t_train = targs.iloc[train_idx]
                f_val = ftrs.iloc[val_idx]
                t_val = targs.iloc[val_idx]
                
                self.tag = f'cv_{fold_idx}'
                self.train(f_train, t_train)
                
                if self.model is None:
                    continue
                
                preds = _predict(self.model, f_val)
                fold_result = self.predict(
                    cycl=f'cv_{fold_idx}',
                    ftrs=f_val,
                    targs=t_val.values.ravel(),
                    ids=[str(i) for i in val_idx],
                    targn='cv_target',
                    what='cross_validate'
                )
                
                scores = self.score(t_val, fold_result['Predicts'])
                fold_scores.append(scores)
                logger.info("CV fold %d %s: %s", fold_idx, code, scores)
                
                del f_train, t_train, f_val, t_val
                gc.collect()
            
            if fold_scores:
                results[code] = {
                    'folds': fold_scores,
                    'mean_accuracy': sum(s.get('accuracy', 0) for s in fold_scores) / len(fold_scores),
                    'mean_logloss': sum(s.get('LogLoss', 0) for s in fold_scores) / len(fold_scores),
                }
        
        logger.info("CV complete: %s", {k: v['mean_accuracy'] for k, v in results.items()})
        return results

    # --------------------------------------------------------------------------------------------------- #
    def evaluate(self, train_targs: DataFrame, valid_targs: DataFrame,
                 runs: list[str] | None = None,
                 vplats: list[str] | None = None) -> Owl:
        """Run full suite of training and predictions.
        
        Args:
            train_targs: Training target data
            valid_targs: Validation target data
            runs: List of algorithm runs (default: all)
            vplats: List of validation platforms (default: all)
        """
        logger.info("Evaluate not yet implemented for this configuration")
        return self

    # --------------------------------------------------------------------------------------------------- #
    def predict(self, cycl: str, ftrs: DataFrame, targs: DataFrame,
                ids: list[Any], targn: str, what: str | None = None,
                model: Any = None, limit: float = 0.8) -> DataFrame:
        """Create predictions from model and score those predictions.
        
        Args:
            cycl: Cycle identifier
            ftrs: Features DataFrame
            targs: Targets DataFrame
            ids: Row identifiers
            targn: Target name
            what: Run type (train/predict/validate)
            model: Optional model override
            limit: LogLoss limit for early stopping
            
        Returns:
            DataFrame with predictions
        """
        if model is None:
            model = self.model
        if model is None:
            logger.error("No model available for prediction")
            return DataFrame()

        result = DataFrame()
        result['ids'] = ids
        result['BinaryTargets'] = targs
        
        predicts = _predict(model, ftrs)
        result['Predicts'] = predicts
        
        result['foldcycl'] = [self.tag for _ in range(len(ids))]
        result['algoname'] = [self.algon for _ in range(len(ids))]
        result['platform'] = [self.plat for _ in range(len(ids))]
        result['runtype'] = [what for _ in range(len(ids))]
        result['targetname'] = [targn for _ in range(len(ids))]

        scores = self.score(targs, result['Predicts'])
        logger.info("Prediction scores: %s", scores)
        
        return result

    # --------------------------------------------------------------------------------------------------- #
    def score(self, targs, predicts) -> dict[str, float]:
        """Score predictions against targets.
        
        Args:
            targs: Target values (array-like)
            predicts: Predicted values — accepts DataFrame from _predict
                      (with 'probability' col) or raw array-like
        
        Returns:
            Dictionary of scoring metrics
        """
        scores: dict[str, float] = {}
        try:
            from sklearn.metrics import accuracy_score, log_loss
            
            # Extract probability values — _predict returns DataFrame with 'probability' col
            if hasattr(predicts, 'columns') and 'probability' in predicts.columns:
                proba = predicts['probability'].values
            elif hasattr(predicts, 'values'):
                proba = predicts.values
            else:
                proba = predicts
            
            if hasattr(model := self.model or [], 'predict_proba'):
                # Binary classification: use class 1 probability
                scores['accuracy'] = float(accuracy_score(targs, (proba >= 0.5).astype(int)))
                scores['LogLoss'] = float(log_loss(targs, proba))
            else:
                preds = proba.ravel() if hasattr(proba, 'ravel') else proba
                scores['accuracy'] = float(accuracy_score(targs, preds))
                scores['LogLoss'] = 0.0
        except Exception as e:
            logger.warning("Scoring failed: %s", e)
            scores['LogLoss'] = 1.0
            scores['accuracy'] = 0.0
        return scores

    # --------------------------------------------------------------------------------------------------- #
    def train(self, ftrs: DataFrame, targs: DataFrame,
              how: str = 'full') -> Owl:
        """Train the model based on the active model configuration.
        
        Args:
            ftrs: Features DataFrame
            targs: Targets DataFrame
            how: Training method ('full' or 'partial')
            
        Returns:
            Self for chaining
        """
        if self.active_model is None:
            logger.error("No active model set. Call setActiveModel() first.")
            return self
            
        algoN = self.active_model['algoN']
        seqN = self.active_model['seqN']
        platN = self.active_model['platN']
        
        TAG = str(uuid.uuid4())[:8]
        targTAG = TAG[-5:]
        trainTAG = TAG[:5]
        
        self.code = self._code(self.tag, algoN, seqN, platN, targTAG)
        
        if not isinstance(targs, DataFrame):
            targs = DataFrame(targs)
            
        self.model = _train(algoN, seqN, platN, ftrs, targs, how, self.model)
        
        del ftrs
        del targs
        gc.collect()
        
        logger.info("Model trained: %s", self.code)
        return self

    # --------------------------------------------------------------------------------------------------- #
    def add_bias(self, adjusters: Any) -> Owl:
        """Apply bias adjustment to model weights.
        
        Args:
            adjusters: Adjustment factor (float, dict, or list)
        """
        logger.info("Bias adjustment not yet implemented")
        return self

    # --------------------------------------------------------------------------------------------------- #
    def _code(self, tag: str, algo: str, seq: str, plat: str, targ: str) -> str:
        """Generate a unique code for this training run."""
        self.algon = f'{algo}_{seq}'
        self.plat = plat
        return f'{tag}_{algo}_{seq}_{plat}_{targ}'

    # --------------------------------------------------------------------------------------------------- #
    def _rtr(self, kind: str, dmap: dict | None = None) -> Owl:
        """Route given inputs into the appropriate algorithm."""
        self.kind = kind
        if dmap is None:
            dmap = self.config.get('dmap', {})
        if isinstance(self.FTRs, str):
            self.dtype = 'STR'
        elif isinstance(self.FTRs, (int, float)):
            self.dtype = 'INT'
        else:
            self.dtype = 'DATA'
        return self


# ====================================================================================================================||
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@||
