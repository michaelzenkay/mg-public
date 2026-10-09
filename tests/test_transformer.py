"""Synthetic aggregation and configuration checks; no data or downloads."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

import mg
import models
import pipelines
from checkpoint import save_ckpt


OPTIONS = dict(num_heads=2, num_layers=2, dropout=0.0, num_view_types=4)
RISK = SimpleNamespace(name='risk', key='risk', horizons=5, weight=1.0)


class SmallEncoder(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.feat_dim = 8
        self.projection = nn.Linear(1, 8)

    def forward(self, images):
        return self.projection(images.mean(dim=(2, 3)))


def batch():
    return (
        torch.randn(12, 4, 1, 8, 8),
        torch.ones(12, 4, dtype=torch.bool),
        torch.arange(4).expand(12, -1),
        torch.tensor([[0.] * 5, [1.] * 5] * 6),
        torch.ones(12, 5, dtype=torch.bool),
        torch.empty(12, 0), torch.empty(12, 0), list(map(str, range(12))),
    )


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.encoder = patch('models.EfficientNetBackbone', SmallEncoder)
        self.encoder.start()
        self.addCleanup(self.encoder.stop)

    def model(self, enabled=True):
        return models.build_model(
            pretrained=False, gpus=[], risk_specs=[RISK],
            use_transformer=enabled, transformer_kwargs=OPTIONS,
        )[0]

    def test_cli_forwards_transformer_configuration(self):
        with patch('sys.argv', ['mg.py', '--config', 'unused.yaml']), \
             patch('pathlib.Path.read_text', return_value=yaml.safe_dump({
                 'ds_csv_path': 'unused.csv', 'use_transformer': True,
                 'transformer_kwargs': OPTIONS,
             })), patch('mg.run_train_staged') as train:
            mg.main()
        self.assertTrue(train.call_args.kwargs['use_transformer'])
        self.assertEqual(train.call_args.kwargs['transformer_kwargs'], OPTIONS)

    def test_training_saves_architecture_before_loading_data(self):
        # Stop at the data boundary: verify the real trainer's saved contract.
        with tempfile.TemporaryDirectory() as tmp, \
             patch('pipelines.train_test_load', side_effect=RuntimeError('data boundary')):
            with self.assertRaisesRegex(RuntimeError, 'data boundary'):
                pipelines.run_train_staged(
                    'unused.csv', results_dir=tmp, use_transformer=True,
                    transformer_kwargs=OPTIONS, target_h=32, target_w=24,
                )
            config = yaml.safe_load(next(Path(tmp).glob('*/run_config.yaml')).read_text())
            self.assertTrue(config['use_transformer'])
            self.assertEqual(config['transformer_kwargs'], OPTIONS)
            self.assertEqual((config['target_h'], config['target_w']), (32, 24))

    def test_risk_training_updates_transformer(self):
        model = self.model()
        before = model.view_transformer.cls_token.detach().clone()
        views_before = model.view_transformer.view_embed.weight.detach().clone()
        loss = pipelines.train_one_epoch_risk(
            model, [batch()], torch.optim.SGD(model.parameters(), lr=.01),
            torch.device('cpu'), risk_specs=[RISK], exam_cat_specs=[],
            exam_reg_specs=[], exam_cat_idx_map={}, exam_reg_idx_map={},
            reg_ranges={}, aux_weight=0., gradient_accumulation=1,
        )
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.view_transformer.cls_token))
        self.assertFalse(torch.equal(views_before, model.view_transformer.view_embed.weight))

    def test_view_identity_and_masking(self):
        model = self.model().eval()
        views, mask, ids, *_ = batch()
        mask[:, -1] = False
        with torch.inference_mode():
            expected = model(views, mask, ids)['risk']
            changed = views.clone(); changed[:, -1] = 999
            torch.testing.assert_close(model(changed, mask, ids)['risk'], expected)
            order = [2, 0, 3, 1]
            torch.testing.assert_close(
                model(views[:, order], mask[:, order], ids[:, order])['risk'], expected,
            )
            self.assertFalse(torch.allclose(model(views, mask, (ids + 1) % 4)['risk'], expected))

    def test_external_eval_reconstructs_both_aggregators(self):
        for enabled in (True, False):
            with self.subTest(transformer=enabled), tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / 'selected.pth'
                model = self.model(enabled).eval()
                save_ckpt(checkpoint, model, None, 0, 0.)
                inputs = batch()
                expected = pipelines.eval_one_epoch_exam(model, [inputs], torch.device('cpu'))
                cfg = {'backbone': 'effv2s', 'risk_specs': [{'name': 'risk', 'horizons': 5}]}
                if enabled:
                    cfg.update(use_transformer=True, transformer_kwargs=OPTIONS)
                # False case deliberately omits the flag to cover legacy configs.
                stdout = io.StringIO()
                with patch('pipelines.pd.read_csv', return_value=pd.DataFrame({'pid_acc': ['sample']})), \
                     patch('pipelines.ExamDataset'), \
                     patch('pipelines.DataLoader', return_value=[inputs]), \
                     contextlib.redirect_stdout(stdout):
                    result = pipelines.eval_external_mirai_dataset(
                        checkpoint, 'unused.csv', gpus=[], run_cfg=cfg,
                    )
                for year in range(1, 6):
                    self.assertEqual(result[f'auc_h{year}yr'], expected[f'auc_h{year}yr'])
                    self.assertIn(f'{year}-year AUC:', stdout.getvalue())
                self.assertAlmostEqual(result['loss'], expected['loss'])
                self.assertNotIn('Mean AUC:', stdout.getvalue())
                if enabled:
                    with self.assertRaises(RuntimeError):
                        self.model(False).load_state_dict(model.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
