"""A corrupt download must never replace the installed checkpoint."""
import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.download_weights import download

class WeightDownloadTests(unittest.TestCase):
    def test_hash_checked_before_replacing_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            (source/'model.pt').write_bytes(b'verified weights')
            dest=root/'checkpoints/model.pt';dest.parent.mkdir();dest.write_bytes(b'installed weights')
            with patch('scripts.download_weights.ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'SHA-256 mismatch'):
                    download('model.pt',dest,base_url=source.as_uri(),sha256='0'*64)
                self.assertEqual(dest.read_bytes(),b'installed weights')
                expected=hashlib.sha256(b'verified weights').hexdigest()
                download('model.pt',dest,base_url=source.as_uri(),sha256=expected)
                self.assertEqual(dest.read_bytes(),b'verified weights')
