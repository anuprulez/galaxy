"""
GTN Search Library - Interface to the GTN SQLite database.

Provides search over Galaxy Training Network tutorials and FAQs
using SQLite FTS5 full-text search with BM25 ranking.
"""

import logging
import re
import sqlite3
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Optional,
)

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

GTN_DATABASE_URL = "https://depot.galaxyproject.org/chatgxy/gtn_search.db"

GTN_VECTOR_STORE_URL = "/home/anup/galaxycodes/galaxy/gtn_vectors/galaxy/lib/galaxy/agents/gtn/vector_store/"

GTN_VECTOR_CHROMADB_URL = "/home/anup/galaxycodes/galaxy/gtn_vectors/galaxy/lib/galaxy/agents/gtn/chromadb/"

log = logging.getLogger(__name__)


def sanitize_fts5_query(query: str, preserve_phrases: bool = True) -> str:
    """Strip FTS5 operators from user input to prevent syntax errors.

    >>> sanitize_fts5_query("climate data, help me analyze (temperature)")
    'climate data help me analyze temperature'
    >>> sanitize_fts5_query('find "exact phrase" in data', preserve_phrases=True)
    'find "exact phrase" in data'
    >>> sanitize_fts5_query('find "exact phrase" in data', preserve_phrases=False)
    'find exact phrase in data'
    """
    if not query or not query.strip():
        return ""

    # Start with the input query
    sanitized = query

    # Replace hyphens with spaces for better matching (RNA-seq -> RNA seq)
    sanitized = sanitized.replace("-", " ")

    # Remove FTS5 special characters that commonly cause syntax errors
    sanitized = sanitized.replace(",", " ")  # Remove commas (column separators)
    sanitized = sanitized.replace("(", " ")  # Remove opening parentheses
    sanitized = sanitized.replace(")", " ")  # Remove closing parentheses
    sanitized = sanitized.replace(":", " ")  # Remove colons (column prefixes)
    sanitized = sanitized.replace("+", " ")  # Remove plus signs (AND operator)
    sanitized = sanitized.replace("?", " ")  # Remove question marks
    sanitized = sanitized.replace("!", " ")  # Remove exclamation marks
    sanitized = sanitized.replace(";", " ")  # Remove semicolons
    sanitized = sanitized.replace("[", " ")  # Remove square brackets
    sanitized = sanitized.replace("]", " ")  # Remove square brackets

    # Handle quotes based on preserve_phrases setting
    if not preserve_phrases:
        sanitized = sanitized.replace('"', " ")  # Remove all quotes
    # else: keep quotes as-is for phrase matching

    # Handle asterisks - remove them unless they're clearly meant for prefix matching
    # Simple heuristic: keep * only if it's at the end of a word
    sanitized = re.sub(r"\*(?!\s|$)", " ", sanitized)  # Remove * not at word boundaries

    # Clean up whitespace: collapse multiple spaces and trim
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    return sanitized


@dataclass
class SearchResult:
    """Represents a search result from the GTN database."""

    id: int
    topic: str
    tutorial: str
    title: str
    url: str
    snippet: str
    score: float
    difficulty: str
    hands_on: bool
    time_estimation: str
    description: str = ""
    result_type: str = "tutorial"  # "tutorial" or "faq"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns only the fields the LLM needs to pick tutorials and
        construct get_tutorial_content calls, keeping token usage low.
        Includes ``score`` so the agent can gauge match quality.
        """
        snippet = self.snippet.replace("<mark>", "").replace("</mark>", "")
        return {
            "title": self.title,
            "topic": self.topic,
            "tutorial": self.tutorial,
            "url": self.url,
            "difficulty": self.difficulty,
            "time_estimation": self.time_estimation,
            "snippet": snippet,
            "score": round(self.score, 2),
        }


@dataclass
class FAQResult:
    """Represents a FAQ search result."""

    id: int
    category: str
    filename: str
    title: str
    area: str
    content: str
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        snippet = self.snippet.replace("<mark>", "").replace("</mark>", "")
        return {
            "title": self.title,
            "category": self.category,
            "area": self.area,
            "snippet": snippet,
            "score": round(self.score, 2),
            "result_type": "faq",
        }


@dataclass
class VectorSearchResult:
    """Represents a search result from vector store."""

    type: str
    topic: str
    score: float
    title: str
    content: str
    url: str
    difficulty: str
    hands_on: bool
    time_estimation: str
    tutorial: str


    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns only the fields the LLM needs to pick tutorials and
        construct get_tutorial_content calls, keeping token usage low.
        Includes ``score`` so the agent can gauge match quality.
        """
        return {
            "title": self.title,
            "topic": self.topic,
            "tutorial": self.tutorial,
            "url": self.url,
            "difficulty": self.difficulty,
            "time_estimation": self.time_estimation,
            "score": round(self.score, 2),
            "content": self.content,
            "hands_on": self.hands_on,
            "type": self.type,
        }
    
@dataclass
class VectorSearchDbResult:
    """Represents a search result from vector store db."""

    type: str
    topic: str
    tutorial: str
    page_content: str
    url: str
    score: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns only the fields the LLM needs to pick tutorials and
        construct get_tutorial_content calls, keeping token usage low.
        Includes ``score`` so the agent can gauge match quality.
        """
        # Truncate page_content to reasonable length for LLM processing
        
        return {
            "type": self.type,
            "topic": self.topic,
            "tutorial": self.tutorial,
            "page_content": self.page_content,
            "score": round(self.score, 2),
            "url": self.url,
            "source": self.source
        }


class GTNSearchDB:
    """Interface to the GTN search database."""

    def __init__(self, db_path: Optional[str] = None, download_url: Optional[str] = None, vector_store_path: Optional[str] = None):
        if db_path is None:
            current_dir = Path(__file__).parent
            self.db_path = current_dir / "data" / "gtn_search.db"
        else:
            self.db_path = Path(db_path)

        self.download_url = download_url or GTN_DATABASE_URL

        if not self.db_path.exists():
            self._download_database()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM tutorials")
                count = cursor.fetchone()[0]
                version = self._read_meta(cursor, "version") or "unknown"
                build_date = self._read_meta(cursor, "build_date") or "unknown"
                log.info(
                    f"GTN database loaded from {self.db_path} "
                    f"(version={version}, built={build_date}, tutorials={count})"
                )
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to initialize GTN database: {e}") from e

    @staticmethod
    def _read_meta(cursor: sqlite3.Cursor, key: str) -> Optional[str]:
        try:
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        except sqlite3.Error:
            return None
        row = cursor.fetchone()
        return row[0] if row else None

    def _download_database(self):
        """Download the GTN database from the configured URL."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.db_path.with_suffix(".db.tmp")
        try:
            log.info(f"GTN database not found locally, downloading from {self.download_url} ...")
            urllib.request.urlretrieve(self.download_url, tmp_path)
            tmp_path.rename(self.db_path)
            log.info(f"GTN database downloaded to {self.db_path}")
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            raise FileNotFoundError(f"GTN database not found at {self.db_path} and download failed: {e}") from e

    def refresh(self) -> None:
        """Force-redownload the database from ``download_url``, replacing any local copy."""
        if self.db_path.exists():
            self.db_path.unlink()
        self._download_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Open a read-only, autocommit connection to the GTN database."""
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def search(
        self,
        query: str,
        limit: int = 5,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        hands_on_only: bool = False,
    ) -> list[SearchResult]:
        """Search tutorials using FTS5 full-text search with optional filters."""
        if not query:
            return []

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                fts_query = sanitize_fts5_query(query, preserve_phrases=True)
                # tutorials_fts columns: title, description, content, topic
                # Weight title/description/topic above raw content so broad
                # queries surface the tutorial that's actually about the topic
                # rather than the one that mentions it most.
                sql = """
                    SELECT
                        t.id,
                        t.topic,
                        t.tutorial,
                        t.title,
                        t.url,
                        t.description,
                        t.difficulty,
                        t.hands_on,
                        t.time_estimation,
                        snippet(tutorials_fts, 2, '<mark>', '</mark>', '...', 30) as snippet,
                        bm25(tutorials_fts, 10.0, 3.0, 1.0, 2.0) as score
                    FROM tutorials_fts
                    JOIN tutorials t ON t.id = tutorials_fts.rowid
                    WHERE tutorials_fts MATCH ?
                """

                params: list[Any] = [fts_query]

                conditions = []
                if topic:
                    conditions.append("t.topic = ?")
                    params.append(topic)

                if difficulty:
                    conditions.append("t.difficulty = ?")
                    params.append(difficulty.lower())

                if hands_on_only:
                    conditions.append("t.hands_on = 1")

                if conditions:
                    sql += " AND " + " AND ".join(conditions)

                sql += " ORDER BY score LIMIT ?"
                params.append(limit)

                results = cursor.execute(sql, params)
                search_results = []
                for row in results:
                    search_results.append(
                        SearchResult(
                            id=row["id"],
                            topic=row["topic"],
                            tutorial=row["tutorial"],
                            title=row["title"],
                            url=row["url"],
                            snippet=row["snippet"],
                            score=abs(row["score"]),  # BM25 scores are negative
                            difficulty=row["difficulty"],
                            hands_on=bool(row["hands_on"]),
                            time_estimation=row["time_estimation"] or "",
                            description=row["description"] or "",
                        )
                    )

                return search_results

        except sqlite3.Error as e:
            log.warning(f"Search failed for query '{query}': {e}")
            return []

    def search_faqs(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        area: Optional[str] = None,
    ) -> list[FAQResult]:
        """Search FAQs using FTS5 full-text search with optional filters."""
        if not query:
            return []

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                fts_query = sanitize_fts5_query(query, preserve_phrases=True)
                # faqs_fts columns: title, content, category, area
                sql = """
                    SELECT
                        f.id,
                        f.category,
                        f.filename,
                        f.title,
                        f.area,
                        f.content,
                        snippet(faqs_fts, 1, '<mark>', '</mark>', '...', 30) as snippet,
                        bm25(faqs_fts, 10.0, 1.0, 2.0, 2.0) as score
                    FROM faqs_fts
                    JOIN faqs f ON f.id = faqs_fts.rowid
                    WHERE faqs_fts MATCH ?
                """

                params: list[Any] = [fts_query]

                conditions = []
                if category:
                    conditions.append("f.category = ?")
                    params.append(category)

                if area:
                    conditions.append("f.area = ?")
                    params.append(area)

                if conditions:
                    sql += " AND " + " AND ".join(conditions)

                sql += " ORDER BY score LIMIT ?"
                params.append(limit)

                results = cursor.execute(sql, params)

                faq_results = []
                for row in results:
                    faq_results.append(
                        FAQResult(
                            id=row["id"],
                            category=row["category"],
                            filename=row["filename"],
                            title=row["title"],
                            area=row["area"] or "",
                            content=row["content"],
                            snippet=row["snippet"],
                            score=abs(row["score"]),
                        )
                    )

                return faq_results

        except sqlite3.Error as e:
            log.warning(f"FAQ search failed for query '{query}': {e}")
            return []
        
    def search_vector_db(
        self,
        query: str,
        limit: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[VectorSearchDbResult]:
        try:
            # Initialize embeddings

            embeddings = OpenAIEmbeddings(
                base_url="https://api.deepinfra.com/v1/openai",
                model="BAAI/bge-large-en-v1.5",
                api_key="",
                tiktoken_enabled=False,
                check_embedding_ctx_length=False
            )

            persist_dir = GTN_VECTOR_CHROMADB_URL
            
            # Check if the persist directory exists
            if not Path(persist_dir).exists():
                log.warning(f"ChromaDB persist directory does not exist: {persist_dir}")
                return []

            vectorstore = Chroma(
                persist_directory=persist_dir,
                collection_name="gtn_tutorials",
                embedding_function=embeddings
            )

            # Use similarity_search_with_score to get relevance scores
            results_with_scores = vectorstore.similarity_search_with_score(query, k=limit)

            log.info(f"Found {len(results_with_scores)} similar documents")

            vector_results = []

            for i, (doc, score) in enumerate(results_with_scores, start=1):
                result = VectorSearchDbResult(
                    type=doc.metadata["type"],
                    topic=doc.metadata["topic"],
                    tutorial=doc.metadata["tutorial"],
                    page_content=str(doc.page_content),
                    score=score,
                    url=doc.metadata["url"],
                    source=doc.metadata["data_source"],
                )
                vector_results.append(result)

            log.info(f"Processed {len(vector_results)} vector DB search results")
            for i, result in enumerate(vector_results[:3]):  # Log first 3 results
                log.debug(f"Result {i+1}: score={result.score:.3f}, source={result.source}")

            return vector_results
            
        except Exception as e:
            log.error(f"Vector DB search failed: {e}")
            return []
    

    def search_vector(
        self,
        query: str,
        limit: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[VectorSearchResult]:
        """Search using vector similarity (RAG retrieval)."""

        self.vector_store_path = GTN_VECTOR_STORE_URL

        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = embed_model

        storage_context = StorageContext.from_defaults(
            persist_dir=self.vector_store_path
        )

        index = load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )

        retriever = index.as_retriever(
            similarity_top_k=limit
        )

        results = retriever.retrieve(query)

        log.info(f"Vector search for query '{query}' returned {len(results)} results")

        vector_results = []

        for i, node_with_score in enumerate(results, start=1):
            node = node_with_score.node
            result = VectorSearchResult(
                type=node.metadata.get("type", ""),
                topic=node.metadata.get("topic", ""),
                score=node_with_score.score,
                title=node.metadata.get("title", ""),
                content=node.get_content(),
                url=node.metadata.get("url", ""),
                difficulty=node.metadata.get("difficulty", ""),
                hands_on=node.metadata.get("hands_on", False),
                time_estimation=node.metadata.get("time_estimation", ""),
                tutorial=node.metadata.get("tutorial", ""),
            )
            vector_results.append(result)
        return vector_results

    def get_tutorial_content(self, topic: str, tutorial: str, max_length: Optional[int] = None) -> Optional[str]:
        """Retrieve tutorial content, optionally truncated to max_length."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                result = cursor.execute(
                    "SELECT content FROM tutorials WHERE topic = ? AND tutorial = ?",
                    (topic, tutorial),
                )

                row = result.fetchone()
                if row:
                    content = row["content"]
                    if max_length and len(content) > max_length:
                        content = content[:max_length] + "..."
                    return content

                return None

        except sqlite3.Error as e:
            log.warning(f"Failed to get tutorial content for {topic}/{tutorial}: {e}")
            return None

    def get_topics(self) -> list[str]:
        """List all available topics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                results = cursor.execute("SELECT DISTINCT topic FROM tutorials ORDER BY topic")

                return [row["topic"] for row in results]

        except sqlite3.Error as e:
            log.warning(f"Failed to get topics: {e}")
            return []

    def get_metadata(self) -> dict[str, Any]:
        """Get database version and build information."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                results = cursor.execute("SELECT key, value FROM metadata")

                metadata = {}
                for row in results:
                    metadata[row["key"]] = row["value"]

                return metadata

        except sqlite3.Error as e:
            log.warning(f"Failed to get metadata: {e}")
            return {}

    def suggest_queries(self, partial: str) -> list[str]:
        """Suggest queries based on partial input."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                results = cursor.execute(
                    "SELECT query FROM example_queries WHERE query LIKE ? LIMIT 5",
                    (f"%{partial}%",),
                )

                suggestions = [row["query"] for row in results]

                if len(suggestions) < 5:
                    topic_results = cursor.execute(
                        """
                        SELECT topic, COUNT(*) as count
                        FROM tutorials
                        WHERE topic LIKE ?
                        GROUP BY topic
                        ORDER BY count DESC
                        LIMIT ?
                        """,
                        (f"%{partial}%", 5 - len(suggestions)),
                    )

                    for row in topic_results:
                        suggestions.append(f"topic:{row['topic']}")

                return suggestions

        except sqlite3.Error as e:
            log.warning(f"Failed to get query suggestions: {e}")
            return []

    def get_tutorial_by_id(self, tutorial_id: int) -> Optional[dict[str, Any]]:
        """Get complete tutorial information by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                result = cursor.execute(
                    """
                    SELECT
                        id, topic, tutorial, title, description, url,
                        difficulty, hands_on, time_estimation, content,
                        questions, objectives, key_points,
                        tools_json, requirements_json, tags_json
                    FROM tutorials
                    WHERE id = ?
                    """,
                    (tutorial_id,),
                )

                row = result.fetchone()
                if row:
                    return dict(row)

                return None

        except sqlite3.Error as e:
            log.warning(f"Failed to get tutorial by ID {tutorial_id}: {e}")
            return None

    def search_by_tools(self, tool_names: list[str], limit: int = 5) -> list[SearchResult]:
        """Search for tutorials that use specific tools."""
        if not tool_names:
            return []

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                tool_conditions = []
                params: list[Any] = []
                for tool in tool_names:
                    tool_conditions.append("tools_json LIKE ?")
                    params.append(f"%{tool}%")

                sql = f"""
                    SELECT
                        id, topic, tutorial, title, url, description,
                        difficulty, hands_on, time_estimation
                    FROM tutorials
                    WHERE {" OR ".join(tool_conditions)}
                    LIMIT ?
                """
                params.append(limit)

                results = cursor.execute(sql, params)

                search_results = []
                for row in results:
                    search_results.append(
                        SearchResult(
                            id=row["id"],
                            topic=row["topic"],
                            tutorial=row["tutorial"],
                            title=row["title"],
                            url=row["url"],
                            snippet=f"Tutorial uses tools: {', '.join(tool_names)}",
                            score=1.0,  # No relevance score for tool search
                            difficulty=row["difficulty"],
                            hands_on=bool(row["hands_on"]),
                            time_estimation=row["time_estimation"] or "",
                            description=row["description"] or "",
                        )
                    )

                return search_results

        except sqlite3.Error as e:
            log.warning(f"Failed to search by tools: {e}")
            return []
