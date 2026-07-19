import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import build


class AutolinkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_registry = build.REGISTRY_PATH
        self.original_metadata = build.METADATA_PATH
        build.REGISTRY_PATH = os.path.join(self.temp.name, "registry.json")
        build.METADATA_PATH = os.path.join(self.temp.name, "metadata.json")
        with open(build.REGISTRY_PATH, "w", encoding="utf-8") as f:
            f.write('{"aliases": {}, "exclude": []}')
        with open(build.METADATA_PATH, "w", encoding="utf-8") as f:
            f.write('{"status_labels": {"synthesis": {"label": "Synthesis", "description": "Test"}}, '
                    '"pages": {"Luxterra": {"status": "synthesis", "source_pages": ["Timeline"], '
                    '"facts": {"Era": "LC"}}}}')
        pages = {
            "Home.md": "# Home\n",
            "Helluvianism.md": "# Helluvianism\n",
            "Luxterra.md": "# Luxterra\n",
            "Timeline.md": "# Timeline\n\n* 670 LC\n> The Great Collapse reshaped [Luxterra](Luxterra).\n\n* Mythic Age\n> Not a dated entry.\n",
        }
        for name, text in pages.items():
            with open(os.path.join(self.temp.name, name), "w", encoding="utf-8") as f:
                f.write(text)
        self.site = build.Site(self.temp.name, os.path.join(self.temp.name, "out"))
        self.site.load()
        helluvian = self.site.by_key["helluvianism"]
        for alias in ["helluvian", "helluvian faith", "helluvian heresy"]:
            self.site.alias_targets[alias] = helluvian

    def tearDown(self):
        build.REGISTRY_PATH = self.original_registry
        build.METADATA_PATH = self.original_metadata
        self.temp.cleanup()

    def test_links_first_occurrence_and_longest_alias(self):
        page = self.site.by_key["luxterra"]
        result = self.site.autolink_html(
            "<p>The Helluvian faith welcomed Helluvian pilgrims and more Helluvian pilgrims.</p>",
            page,
        )
        self.assertIn('class="autolink">Helluvian faith</a>', result)
        self.assertEqual(result.count('href="helluvianism.html"'), 1)

    def test_skips_existing_links_headings_and_code(self):
        page = self.site.by_key["luxterra"]
        result = self.site.autolink_html(
            '<h2>Helluvian</h2><p><a href="x.html">Helluvian</a> '
            '<code>Helluvian</code> Helluvian</p>', page)
        self.assertEqual(result.count('href="helluvianism.html"'), 1)
        self.assertIn('<h2>Helluvian</h2>', result)
        self.assertIn('<code>Helluvian</code>', result)

    def test_never_self_links(self):
        page = self.site.by_key["helluvianism"]
        result = self.site.autolink_html("<p>Helluvianism and Helluvian faith.</p>", page)
        self.assertNotIn("autolink", result)

    def test_extracts_dated_timeline_events_without_inventing_dates(self):
        events = self.site.timeline_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["date"], "670 LC")
        self.assertIn('href="luxterra.html"', events[0]["html"])
        self.assertIn("Great Collapse", events[0]["text"])

    def test_loads_reviewed_metadata_without_inferring_facts(self):
        page = self.site.by_key["luxterra"]
        self.assertEqual(page.metadata["status"], "synthesis")
        self.assertEqual(page.metadata["facts"], {"Era": "LC"})


if __name__ == "__main__":
    unittest.main()
