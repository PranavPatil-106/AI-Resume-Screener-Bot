import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.hf_store import HuggingFaceVectorStore

class TestFixes(unittest.TestCase):
    def setUp(self):
        self.store = HuggingFaceVectorStore()
        # Mock the actual model to avoid downloading/calling API during quick test
        self.store.model = MagicMock()
        self.store.model.encode.return_value = np.random.rand(2, 384).astype(np.float32) # Mock batch return
        self.store.use_hf_api = False

    def test_embed_batch_local(self):
        texts = ["text1", "text2"]
        embeddings = self.store.embed_batch(texts)
        self.assertEqual(len(embeddings), 2)
        # Verify encode was called with the list
        self.store.model.encode.assert_called_with(texts)

    def test_add_documents_batch(self):
        docs = [
            {"resume_id": "1", "content": "resume content 1"},
            {"resume_id": "2", "content": "resume content 2"}
        ]
        self.store.add_documents(docs)
        self.assertEqual(len(self.store.vectors), 2)
        self.assertEqual(len(self.store.documents), 2)

if __name__ == '__main__':
    unittest.main()
