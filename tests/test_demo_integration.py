"""Check labels retained when integrating global and regional demos."""
import importlib.util
import unittest
from unittest.mock import patch, Mock


@unittest.skipUnless(importlib.util.find_spec('plotly'), 'install demo dependencies')
class DemoIntegrationTests(unittest.TestCase):
    def test_global_raw_fallback_is_not_labelled_calibrated(self):
        from demo.app import _format_markdown, _result_html
        result = dict(mode='global', box=(0, 1, 0, 1), verdict='ABSTAIN',
                      quantile=None, temperature_fitted=False,
                      conformal_fitted=False, calibration_matches_checkpoint=False,
                      lat=0.5, lon=0.5, cell_id=0, cal_conf=0.2, raw_conf=0.2,
                      T=1, set_size=1, K=300)
        details = _format_markdown(result)
        self.assertIn('Uncalibrated softmax confidence', details)
        self.assertIn('no conformal guarantee', details)
        summary = _result_html(result)
        self.assertIn('Raw model score', summary)
        self.assertNotIn('Calibrated confidence', summary)

    def test_kerala_never_claims_conformal_or_temperature_fit(self):
        from demo.app import _format_markdown
        details = _format_markdown(dict(mode='kerala', box=(8, 13, 74, 78),
            lat=10, lon=76, cell_id=2, raw_conf=0.8))
        self.assertIn('uncalibrated', details)
        self.assertIn('no ACCEPT/ABSTAIN threshold', details)

    def test_verified_global_calibration_is_labelled(self):
        from demo.app import _format_markdown
        details = _format_markdown(dict(mode='global', box=(0, 1, 0, 1),
            verdict='ABSTAIN', quantile=0.99, temperature_fitted=True,
            conformal_fitted=True, calibration_matches_checkpoint=True,
            lat=0.5, lon=0.5, cell_id=0, cal_conf=0.2, raw_conf=0.8,
            T=3.5, set_size=150, K=300))
        self.assertIn('checkpoint verified', details)
        self.assertIn('150 of 300', details)

    @unittest.skipUnless(importlib.util.find_spec('gradio'), 'install gradio')
    def test_gradio6_launcher_preserves_classroom_styling(self):
        from demo.app import launch_demo, DEMO_CSS
        ui = Mock()
        with patch('demo.app.build_ui', return_value=ui), patch('gradio.__version__', '6.0.0'):
            launch_demo(server_name='127.0.0.1')
        self.assertEqual(ui.launch.call_args.kwargs['css'], DEMO_CSS)
        self.assertIn('theme', ui.launch.call_args.kwargs)
        self.assertEqual(ui.launch.call_args.kwargs['server_name'], '127.0.0.1')
