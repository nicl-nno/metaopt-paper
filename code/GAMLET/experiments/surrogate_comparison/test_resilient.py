import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import resilient


class PersistenceTest(unittest.TestCase):
    def test_locked_destination_retries_without_changing_json(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'result.json'
            original=resilient.os.replace
            calls=[]
            def intermittent(source,destination):
                calls.append(1)
                if len(calls)<3:raise PermissionError('simulated transient Windows sharing violation')
                original(source,destination)
            with patch.object(resilient.os,'replace',side_effect=intermittent), patch.object(resilient.time,'sleep'):
                resilient.save(target,{'value':.123,'scores':[1,2,3]})
            self.assertEqual(target.read_text(encoding='utf-8'),json.dumps({'value':.123,'scores':[1,2,3]},indent=2))
            self.assertEqual(len(calls),3)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_permanent_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'result.json'
            target.write_text('previous result')
            with patch.object(resilient.os,'replace',side_effect=PermissionError('persistent denial')), patch.object(resilient.time,'sleep'):
                with self.assertRaises(PermissionError):resilient.save(target,{'value':1})
            self.assertEqual(target.read_text(),'previous result')


if __name__=='__main__':unittest.main()
