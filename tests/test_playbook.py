"""Tests for Playbook functionality including persistence."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from ace import Playbook, DeltaBatch, DeltaOperation


class TestPlaybook(unittest.TestCase):
    """Test Playbook class functionality."""

    def setUp(self):
        """Set up test playbook with sample data."""
        self.playbook = Playbook()

        # Add test bullets
        self.bullet1 = self.playbook.add_bullet(
            section="general",
            content="Always be clear",
            metadata={"helpful": 5, "harmful": 0}
        )

        self.bullet2 = self.playbook.add_bullet(
            section="math",
            content="Show your work",
            metadata={"helpful": 3, "harmful": 1}
        )

    def test_add_bullet(self):
        """Test adding bullets to playbook."""
        bullet = self.playbook.add_bullet(
            section="test",
            content="Test content"
        )

        self.assertIsNotNone(bullet)
        self.assertEqual(bullet.section, "test")
        self.assertEqual(bullet.content, "Test content")
        self.assertEqual(len(self.playbook.bullets()), 3)

    def test_update_bullet(self):
        """Test updating existing bullet."""
        updated = self.playbook.update_bullet(
            self.bullet1.id,
            content="Updated content",
            metadata={"helpful": 10}
        )

        self.assertIsNotNone(updated)
        self.assertEqual(updated.content, "Updated content")
        self.assertEqual(updated.helpful, 10)

    def test_tag_bullet(self):
        """Test tagging bullets."""
        self.playbook.tag_bullet(self.bullet1.id, "helpful", 2)
        bullet = self.playbook.get_bullet(self.bullet1.id)

        self.assertEqual(bullet.helpful, 7)  # 5 + 2

    def test_remove_bullet(self):
        """Test removing bullets."""
        self.playbook.remove_bullet(self.bullet1.id)

        self.assertIsNone(self.playbook.get_bullet(self.bullet1.id))
        self.assertEqual(len(self.playbook.bullets()), 1)

    def test_apply_delta(self):
        """Test applying delta operations."""
        delta = DeltaBatch(
            reasoning="Test delta operations",
            operations=[
                DeltaOperation(
                    type="ADD",
                    section="new",
                    content="New strategy"
                ),
                DeltaOperation(
                    type="UPDATE",
                    section="",  # Section is required but not used for UPDATE
                    bullet_id=self.bullet1.id,
                    content="Modified content"
                ),
                DeltaOperation(
                    type="TAG",
                    section="",  # Section is required but not used for TAG
                    bullet_id=self.bullet2.id,
                    metadata={"harmful": 2}
                ),
            ]
        )

        self.playbook.apply_delta(delta)

        # Check ADD operation
        self.assertEqual(len(self.playbook.bullets()), 3)

        # Check UPDATE operation
        bullet1 = self.playbook.get_bullet(self.bullet1.id)
        self.assertEqual(bullet1.content, "Modified content")

        # Check TAG operation
        bullet2 = self.playbook.get_bullet(self.bullet2.id)
        self.assertEqual(bullet2.harmful, 3)  # 1 + 2

    def test_dumps_loads(self):
        """Test JSON serialization and deserialization."""
        # Serialize
        json_str = self.playbook.dumps()
        self.assertIsInstance(json_str, str)

        # Verify it's valid JSON
        data = json.loads(json_str)
        self.assertIn("bullets", data)
        self.assertIn("sections", data)

        # Deserialize
        loaded = Playbook.loads(json_str)

        # Verify content matches
        self.assertEqual(len(loaded.bullets()), len(self.playbook.bullets()))

        for original, loaded_bullet in zip(self.playbook.bullets(), loaded.bullets()):
            self.assertEqual(original.id, loaded_bullet.id)
            self.assertEqual(original.content, loaded_bullet.content)
            self.assertEqual(original.helpful, loaded_bullet.helpful)

    def test_save_to_file(self):
        """Test saving playbook to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save playbook
            self.playbook.save_to_file(temp_path)

            # Verify file exists
            self.assertTrue(os.path.exists(temp_path))

            # Verify content is valid JSON
            with open(temp_path, 'r') as f:
                data = json.load(f)

            self.assertIn("bullets", data)
            self.assertIn("sections", data)
            self.assertEqual(len(data["bullets"]), 2)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_from_file(self):
        """Test loading playbook from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save original
            self.playbook.save_to_file(temp_path)

            # Load from file
            loaded = Playbook.load_from_file(temp_path)

            # Verify content matches
            self.assertEqual(len(loaded.bullets()), 2)

            loaded_bullets = {b.id: b for b in loaded.bullets()}
            original_bullets = {b.id: b for b in self.playbook.bullets()}

            for bid, original in original_bullets.items():
                loaded_bullet = loaded_bullets[bid]
                self.assertEqual(loaded_bullet.content, original.content)
                self.assertEqual(loaded_bullet.section, original.section)
                self.assertEqual(loaded_bullet.helpful, original.helpful)
                self.assertEqual(loaded_bullet.harmful, original.harmful)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_creates_parent_dirs(self):
        """Test that save_to_file creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "dir", "playbook.json")

            # Parent dirs don't exist yet
            self.assertFalse(os.path.exists(os.path.dirname(nested_path)))

            # Save should create them
            self.playbook.save_to_file(nested_path)

            # Verify file was created
            self.assertTrue(os.path.exists(nested_path))

            # Verify we can load it back
            loaded = Playbook.load_from_file(nested_path)
            self.assertEqual(len(loaded.bullets()), 2)

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as context:
            Playbook.load_from_file("nonexistent_file.json")

        self.assertIn("not found", str(context.exception))

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises appropriate error."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write("not valid json {")
            temp_path = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                Playbook.load_from_file(temp_path)
        finally:
            os.remove(temp_path)

    def test_as_prompt(self):
        """Test playbook prompt generation."""
        prompt = self.playbook.as_prompt()

        self.assertIn("general", prompt)
        self.assertIn("math", prompt)
        self.assertIn("Always be clear", prompt)
        self.assertIn("Show your work", prompt)
        self.assertIn("helpful=5", prompt)

    def test_stats(self):
        """Test playbook statistics."""
        stats = self.playbook.stats()

        self.assertEqual(stats["sections"], 2)
        self.assertEqual(stats["bullets"], 2)
        self.assertEqual(stats["tags"]["helpful"], 8)  # 5 + 3
        self.assertEqual(stats["tags"]["harmful"], 1)
        self.assertEqual(stats["tags"]["neutral"], 0)

    def test_empty_playbook_serialization(self):
        """Test that empty playbook can be saved and loaded."""
        empty = Playbook()

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            # Save empty playbook
            empty.save_to_file(temp_path)

            # Load it back
            loaded = Playbook.load_from_file(temp_path)

            # Verify it's empty
            self.assertEqual(len(loaded.bullets()), 0)
            self.assertEqual(loaded.stats()["bullets"], 0)
            self.assertEqual(loaded.stats()["sections"], 0)

        finally:
            os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()