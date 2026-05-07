#!/usr/bin/env python3
"""
Build a persistent GTN vector database with Chroma.

This script processes the Galaxy Training Network repository and creates
an embedded vector database using LangChain + Chroma for persistent storage.

Usage:
    python build_vector_database.py <gtn_repo_path> [--output <db_path>]
"""

import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Optional, Union

# Suppress HuggingFace Hub warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """Embedding wrapper using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


VECTOR_DATABASE_VERSION = "1.0.0"


@dataclass
class FAQ:
    """Represents a GTN FAQ entry."""

    category: str  # "galaxy" or "gtn"
    filename: str
    title: str
    area: str = ""
    box_type: str = ""
    content: str = ""
    content_hash: str = ""
    last_modified: str = ""
    contributors: list[str] = field(default_factory=list)


@dataclass
class Tutorial:
    """Represents a GTN tutorial with all its metadata."""

    topic: str
    tutorial: str
    title: str
    description: str = ""
    url: str = ""
    difficulty: str = "intermediate"
    hands_on: bool = True
    time_estimation: str = ""
    content: str = ""
    questions: str = ""
    objectives: str = ""
    key_points: str = ""
    tools: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    content_hash: str = ""
    last_modified: str = ""
    gtn_commit: str = ""
    zenodo_link: str = ""
    workflows: list[str] = field(default_factory=list)


class GTNVectorDatabaseBuilder:
    """Builds a persistent GTN vector database with Chroma."""

    def __init__(self, gtn_path: Path, output_path: Optional[Path] = None, collection_name: str = "gtn_vector_database"):
        self.gtn_path = gtn_path
        self.output_path = output_path or (Path(__file__).parent / "data" / "gtn_vector_database")
        self.collection_name = collection_name
        self.tutorials: list[Tutorial] = []
        self.faqs: list[FAQ] = []
        self.documents: list[dict[str, Any]] = []

    def build(self):
        """Build the complete vector database."""
        log.info(f"Building GTN vector database from {self.gtn_path}")

        self.output_path.mkdir(parents=True, exist_ok=True)

        self.collect_tutorials()
        self.collect_faqs()
        self.create_documents()
        self.create_vector_database()

        log.info(f"Vector database built successfully: {self.output_path}")
        log.info(f"Total documents indexed: {len(self.documents)}")

    def collect_tutorials(self):
        """Collect all tutorials from the GTN repository."""
        topics_dir = self.gtn_path / "topics"

        if not topics_dir.exists():
            raise ValueError(f"Topics directory not found: {topics_dir}")

        for topic_dir in topics_dir.iterdir():
            if not topic_dir.is_dir() or topic_dir.name.startswith("."):
                continue

            topic = topic_dir.name
            tutorials_dir = topic_dir / "tutorials"
            if not tutorials_dir.exists():
                continue

            for tutorial_dir in tutorials_dir.iterdir():
                if not tutorial_dir.is_dir():
                    continue

                tutorial_name = tutorial_dir.name
                tutorial_file = tutorial_dir / "tutorial.md"
                if tutorial_file.exists():
                    try:
                        tutorial = self.parse_tutorial(tutorial_file, topic, tutorial_name)
                        if tutorial:
                            self.tutorials.append(tutorial)
                            log.debug(f"Parsed tutorial: {topic}/{tutorial_name}")
                    except (OSError, ValueError, KeyError) as e:
                        log.warning(f"Failed to parse {topic}/{tutorial_name}: {e}")

    def collect_faqs(self):
        """Collect all FAQs from the GTN repository."""
        faqs_dir = self.gtn_path / "faqs"

        if not faqs_dir.exists():
            log.warning(f"FAQs directory not found: {faqs_dir}")
            return

        galaxy_faqs_dir = faqs_dir / "galaxy"
        if galaxy_faqs_dir.exists():
            for faq_file in galaxy_faqs_dir.glob("*.md"):
                if faq_file.name != "index.md":
                    try:
                        faq = self.parse_faq(faq_file, "galaxy")
                        if faq:
                            self.faqs.append(faq)
                            log.debug(f"Parsed FAQ: galaxy/{faq_file.name}")
                    except (OSError, ValueError, KeyError) as e:
                        log.warning(f"Failed to parse FAQ galaxy/{faq_file.name}: {e}")

        gtn_faqs_dir = faqs_dir / "gtn"
        if gtn_faqs_dir.exists():
            for faq_file in gtn_faqs_dir.glob("*.md"):
                if faq_file.name != "index.md":
                    try:
                        faq = self.parse_faq(faq_file, "gtn")
                        if faq:
                            self.faqs.append(faq)
                            log.debug(f"Parsed FAQ: gtn/{faq_file.name}")
                    except (OSError, ValueError, KeyError) as e:
                        log.warning(f"Failed to parse FAQ gtn/{faq_file.name}: {e}")

    def parse_faq(self, faq_file: Path, category: str) -> Optional[FAQ]:
        """Parse a FAQ markdown file, returning None on failure."""
        try:
            with open(faq_file, encoding="utf-8") as f:
                content = f.read()

            frontmatter = {}
            if content.startswith("---"):
                try:
                    end_index = content.index("---", 3)
                    yaml_content = content[3:end_index]
                    frontmatter = self.parse_yaml_simple(yaml_content)
                    content = content[end_index + 3 :].strip()
                except ValueError:
                    pass

            stat = faq_file.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            content_bytes = content.encode("utf-8")
            content_hash = md5(content_bytes).hexdigest()

            contributors = []
            if "contributors" in frontmatter:
                contrib_value = frontmatter["contributors"]
                if isinstance(contrib_value, str):
                    contrib_str = contrib_value.strip("[]")
                    contributors = [c.strip() for c in contrib_str.split(",") if c.strip()]
                elif isinstance(contrib_value, list):
                    contributors = contrib_value

            faq = FAQ(
                category=category,
                filename=faq_file.stem,
                title=frontmatter.get("title", faq_file.stem.replace("-", " ").title()),
                area=frontmatter.get("area", ""),
                box_type=frontmatter.get("box_type", ""),
                content=content,
                content_hash=content_hash,
                last_modified=last_modified,
                contributors=contributors,
            )

            return faq

        except (OSError, ValueError, KeyError) as e:
            log.warning(f"Error parsing FAQ {faq_file}: {e}")
            return None

    def parse_tutorial(self, tutorial_file: Path, topic: str, tutorial_name: str) -> Optional[Tutorial]:
        """Parse a tutorial markdown file, returning None on failure."""
        try:
            with open(tutorial_file, encoding="utf-8") as f:
                content = f.read()

            frontmatter = {}
            if content.startswith("---"):
                try:
                    end_index = content.index("---", 3)
                    yaml_content = content[3:end_index]
                    frontmatter = self.parse_yaml_simple(yaml_content)
                    content = content[end_index + 3 :].strip()
                except ValueError:
                    pass

            questions = self.extract_section(content, "questions")
            objectives = self.extract_section(content, "objectives")
            key_points = self.extract_section(content, "keypoints") or self.extract_section(content, "key_points")

            base_url = "https://training.galaxyproject.org/training-material"
            url = f"{base_url}/topics/{topic}/tutorials/{tutorial_name}/tutorial.html"

            stat = tutorial_file.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            content_bytes = content.encode("utf-8")
            content_hash = md5(content_bytes).hexdigest()

            tutorial = Tutorial(
                topic=topic,
                tutorial=tutorial_name,
                title=frontmatter.get("title", tutorial_name.replace("-", " ").title()),
                description=frontmatter.get("description", ""),
                url=url,
                difficulty=str(frontmatter.get("level", "intermediate")).lower(),
                hands_on=frontmatter.get("hands_on", True) not in (False, "false", "False"),
                time_estimation=frontmatter.get("time_estimation", ""),
                content=content[:50000],
                questions=questions,
                objectives=objectives,
                key_points=key_points,
                tools=self.extract_list(frontmatter.get("tools", [])),
                requirements=self.extract_list(frontmatter.get("requirements", [])),
                tags=self.extract_list(frontmatter.get("tags", [])),
                content_hash=content_hash,
                last_modified=last_modified,
                zenodo_link=str(frontmatter.get("zenodo_link", "") or ""),
            )

            return tutorial

        except (OSError, ValueError, KeyError) as e:
            log.warning(f"Error parsing tutorial {tutorial_file}: {e}")
            return None

    def parse_yaml_simple(self, yaml_content: str) -> dict[str, Any]:
        """Simple YAML frontmatter parser (no external dependencies)."""
        result: dict[str, Any] = {}
        current_list: Optional[list[str]] = None

        for line in yaml_content.split("\n"):
            line = line.rstrip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("  - ") or line.startswith("- "):
                if current_list is not None:
                    item = line.strip("- ").strip()
                    if item:
                        current_list.append(item)
                continue

            if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                parts = line.split(":", 1)
                key = parts[0].strip()
                str_value = parts[1].strip() if len(parts) > 1 else ""
                was_quoted = False
                if str_value.startswith('"') and str_value.endswith('"'):
                    str_value = str_value[1:-1]
                    was_quoted = True
                elif str_value.startswith("'") and str_value.endswith("'"):
                    str_value = str_value[1:-1]
                    was_quoted = True

                if str_value.lower() == "true":
                    parsed_value = True
                elif str_value.lower() == "false":
                    parsed_value = False
                elif str_value == "" and not was_quoted:
                    current_list = []
                    result[key] = current_list
                    continue
                else:
                    parsed_value = str_value

                result[key] = parsed_value
                current_list = None

        return result

    def extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from markdown content."""
        escaped = re.escape(section_name)
        patterns = [
            rf"{{:\s*\.{escaped}.*?}}(.*?)(?:{{:|^#|\Z)",
            rf">{{%\s*icon\s+{escaped}.*?%}}.*?\n(.*?)(?:^#|\Z)",
            rf"<{escaped}.*?>(.*?)</{escaped}>",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
                text = re.sub(r"{%.*?%}", "", text)
                text = re.sub(r"{:.*?}", "", text)
                return text.strip()[:1000]

        return ""

    def extract_list(self, value: Any) -> list[str]:
        """Extract a list from various input formats."""
        if isinstance(value, list):
            return [str(item) for item in value]
        elif isinstance(value, str):
            return [value]
        return []

    def create_documents(self):
        """Create document payloads from tutorials and FAQs."""
        for tutorial in self.tutorials:
            text_parts = [
                tutorial.title,
                tutorial.description,
                tutorial.content,
                tutorial.questions,
                tutorial.objectives,
                tutorial.key_points,
            ]
            text = "\n".join(part for part in text_parts if part)

            metadata = {
                "type": "tutorial",
                "topic": tutorial.topic,
                "tutorial": tutorial.tutorial,
                "title": tutorial.title,
                "url": tutorial.url,
                "difficulty": tutorial.difficulty,
                "hands_on": str(tutorial.hands_on),
                "time_estimation": tutorial.time_estimation,
                "tools": json.dumps(tutorial.tools),
                "requirements": json.dumps(tutorial.requirements),
                "tags": json.dumps(tutorial.tags),
                "content_hash": tutorial.content_hash,
                "last_modified": tutorial.last_modified,
                "zenodo_link": tutorial.zenodo_link,
            }

            self.documents.append({
                "id": f"tutorial_{tutorial.topic}_{tutorial.tutorial}",
                "text": text,
                "metadata": metadata,
            })

        for faq in self.faqs:
            text = f"{faq.title}\n{faq.content}"
            metadata = {
                "type": "faq",
                "category": faq.category,
                "filename": faq.filename,
                "title": faq.title,
                "area": faq.area,
                "box_type": faq.box_type,
                "content_hash": faq.content_hash,
                "last_modified": faq.last_modified,
                "contributors": json.dumps(faq.contributors),
            }

            self.documents.append({
                "id": f"faq_{faq.category}_{faq.filename}",
                "text": text,
                "metadata": metadata,
            })

    def create_vector_database(self):
        """Create and persist the Chroma vector database."""
        if Chroma is None:
            raise ImportError(
                "Chroma vector store support is not available. Install chromadb and langchain with Chroma support."
            )

        persist_dir = self.output_path
        persist_dir.mkdir(parents=True, exist_ok=True)

        if HuggingFaceEmbeddings is not None:
            embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            log.info("HuggingFaceEmbeddings unavailable, falling back to sentence-transformers directly.")
            embed_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        texts = [doc["text"] for doc in self.documents]
        metadatas = [doc["metadata"] for doc in self.documents]
        ids = [doc["id"] for doc in self.documents]

        vector_db = Chroma.from_texts(
            texts,
            embed_model,
            metadatas=metadatas,
            ids=ids,
            persist_directory=str(persist_dir),
            collection_name=self.collection_name,
        )
        vector_db.persist()

        log.info(f"Chroma vector database created with {len(self.documents)} documents")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build GTN Vector Database from GTN Repository")
    parser.add_argument("gtn_repo_path", type=str, help="Path to cloned GTN repository")
    parser.add_argument("--output", type=str, help="Output Chroma database path (default: data/gtn_vector_database relative to this script)")
    parser.add_argument("--collection-name", type=str, default="gtn_vector_database", help="Chroma collection name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (debug) logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    gtn_path = Path(args.gtn_repo_path)
    output_path = Path(args.output) if args.output else None

    if not gtn_path.exists():
        log.error(f"GTN repository path does not exist: {gtn_path}")
        sys.exit(1)

    try:
        builder = GTNVectorDatabaseBuilder(gtn_path, output_path, args.collection_name)
        builder.build()
    except Exception as e:
        log.error(f"Failed to build vector database: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
