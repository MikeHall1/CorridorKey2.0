import json
import tempfile
import unittest
from pathlib import Path

from corridorkey.benchmark_report import export_report, read_report, report_view


class PairedReportTests(unittest.TestCase):
    def report(self, complete):
        sample = {'schema_version': 1, 'kind': 'inference', 'summary': {'success': True},
                  'inference': {'status': 'measured', 'quality': None},
                  'runs': [{'elapsed_seconds': 1, 'jobs': [{'status': 'complete'}]}]}
        pairs = [{'index': index, 'order': ['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline'],
                  'samples': {'baseline': sample, 'candidate': sample}} for index in range(3)]
        if not complete:
            pairs = [dict(pairs[0], samples={'baseline': sample})]
        return {'schema_version': 1, 'kind': 'inference_pair', 'status': 'complete' if complete else 'cancelled',
                'summary': {'success': complete, 'speedup': 1 if complete else None}, 'pairs': pairs,
                'test_fixture': True}

    def test_completed_and_partial_pair_reports_inspect_and_export_exact_bytes(self):
        for complete in (True, False):
            with self.subTest(complete=complete), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                path, target = root / 'report.json', root / 'copy.json'
                data = self.report(complete)
                path.write_text(json.dumps(data))
                report, raw = read_report(path)
                self.assertEqual(report, data)
                self.assertIn('inference_pair', report_view(report))
                self.assertEqual(report['summary']['speedup'], 1 if complete else None)
                export_report(path, target)
                self.assertEqual(target.read_bytes(), raw)

    def test_failed_partial_cannot_be_inspected_as_completed_speedup(self):
        for defect in ('speedup', 'status', 'pairs', 'order', 'missing_side', 'failed_sample'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as folder:
                data = self.report(defect in ('pairs', 'order', 'missing_side', 'failed_sample'))
                if defect == 'speedup':
                    data['summary']['speedup'] = 2
                elif defect == 'status':
                    data['status'] = 'complete'
                elif defect == 'pairs':
                    data['pairs'] = []
                elif defect == 'order':
                    data['pairs'][1]['order'].reverse()
                elif defect == 'failed_sample':
                    data['pairs'][0]['samples']['candidate']['summary']['success'] = False
                else:
                    del data['pairs'][0]['samples']['candidate']
                path = Path(folder) / 'report.json'
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    read_report(path)

    def test_malformed_nested_pair_nodes_raise_domain_errors_before_export(self):
        for field, value in (('summary', None), ('summary', []), ('inference', 42),
                             ('inference', []), ('runs', [42]), ('runs', [None]),
                             ('runs', [{'elapsed_seconds': 1, 'jobs': [42]}])):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as folder:
                data = self.report(True)
                data['pairs'][0]['samples']['candidate'][field] = value
                source, target = Path(folder) / 'report.json', Path(folder) / 'export.json'
                source.write_text(json.dumps(data))
                original = source.read_bytes()
                with self.assertRaises(ValueError):
                    read_report(source)
                with self.assertRaises(ValueError):
                    export_report(source, target)
                self.assertFalse(target.exists())
                self.assertEqual(source.read_bytes(), original)
