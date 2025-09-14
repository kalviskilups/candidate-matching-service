"""
Weaviate hybrid search engine with BM25 + vector fusion.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import HybridFusion

from api.services.embedding import embedding_service

logger = logging.getLogger(__name__)


class WeaviateHybridSearchEngine:
    """
    Weaviate-native hybrid search engine using built-in BM25 + vector fusion.

    Uses Weaviate's native hybrid search with configurable fusion methods:
    - Relative Score Fusion (default): Combines normalized scores
    - Ranked Fusion: Combines based on rankings
    """

    def __init__(
        self,
        weaviate_url: Optional[str] = None,
        fusion_method: str = "relative_score",
    ):
        """
        Initialize the Weaviate hybrid search engine.

        Args:
            weaviate_url: URL of the Weaviate instance
            fusion_method: "relative_score" or "ranked" fusion
        """
        self.weaviate_url = weaviate_url or os.getenv(
            "WEAVIATE_URL", "http://localhost:8080"
        )
        self.client: Optional[weaviate.WeaviateClient] = None
        self.jobs_class_name = "Job"
        self.candidates_class_name = "Candidate"
        self.vector_dimension = int(os.getenv("VECTOR_DIMENSION", "384"))
        if fusion_method == "relative_score":
            self.fusion_type = HybridFusion.RELATIVE_SCORE
        elif fusion_method == "ranked":
            self.fusion_type = HybridFusion.RANKED
        else:
            raise ValueError(
                f"Unknown fusion method: {fusion_method}. "
                f"Use 'relative_score' or 'ranked'"
            )

        self.fusion_method_name = fusion_method

    async def initialize(self) -> None:
        """
        Initialize connection to Weaviate and create schemas.

        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger.info(f"Connecting to Weaviate at {self.weaviate_url}")

            url_parts = self.weaviate_url.replace("http://", "").replace("https://", "")
            if ":" in url_parts:
                host, port = url_parts.split(":")
                port = int(port)
            else:
                host = url_parts
                port = 8080

            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                additional_config=weaviate.classes.init.AdditionalConfig(
                    timeout=weaviate.classes.init.Timeout(init=30, query=30, insert=60)
                ),
            )

            if not self.client.is_ready():
                raise RuntimeError("Weaviate is not ready")

            await self._create_schemas()

            logger.info(
                f"Weaviate hybrid search engine initialized with "
                f"{self.fusion_method_name} fusion"
            )

        except Exception as e:
            raise RuntimeError(f"Weaviate initialization failed: {e}")

    async def _create_schemas(self) -> None:
        """
        Create Weaviate schemas for jobs and candidates with hybrid search support.
        """
        collections = self.client.collections

        job_properties = [
            Property(name="job_id", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="skills", data_type=DataType.TEXT_ARRAY),
            Property(name="location_city", data_type=DataType.TEXT),
            Property(name="location_country", data_type=DataType.TEXT),
            Property(name="employment_type", data_type=DataType.TEXT),
            Property(name="seniority", data_type=DataType.TEXT),
            Property(name="language", data_type=DataType.TEXT),
            Property(name="salary_min", data_type=DataType.INT),
            Property(name="salary_max", data_type=DataType.INT),
            Property(name="remote", data_type=DataType.BOOL),
            Property(name="created_at", data_type=DataType.TEXT),
            Property(name="searchable_text", data_type=DataType.TEXT),
        ]

        candidate_properties = [
            Property(name="candidate_id", data_type=DataType.TEXT),
            Property(name="name", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="summary", data_type=DataType.TEXT),
            Property(name="skills", data_type=DataType.TEXT_ARRAY),
            Property(name="years_experience", data_type=DataType.INT),
            Property(name="location_city", data_type=DataType.TEXT),
            Property(name="location_country", data_type=DataType.TEXT),
            Property(name="languages", data_type=DataType.TEXT_ARRAY),
            Property(name="open_to_remote", data_type=DataType.BOOL),
            Property(name="desired_salary", data_type=DataType.INT),
            Property(name="right_to_work_eu", data_type=DataType.BOOL),
            Property(name="last_updated", data_type=DataType.TEXT),
            Property(name="searchable_text", data_type=DataType.TEXT),
        ]

        if collections.exists(self.jobs_class_name):
            collections.delete(self.jobs_class_name)
            logger.info(f"Deleted existing {self.jobs_class_name} collection")

        collections.create(
            name=self.jobs_class_name,
            properties=job_properties,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            inverted_index_config=Configure.inverted_index(),
        )
        logger.info(
            f"Created {self.jobs_class_name} collection with hybrid search support"
        )

        if collections.exists(self.candidates_class_name):
            collections.delete(self.candidates_class_name)
            logger.info(f"Deleted existing {self.candidates_class_name} collection")

        collections.create(
            name=self.candidates_class_name,
            properties=candidate_properties,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                )
            ),
            inverted_index_config=Configure.inverted_index(),
        )
        logger.info(
            f"Created {self.candidates_class_name} collection with "
            f"hybrid search support"
        )

    async def index_jobs(self, jobs_data: List[Tuple[Dict, np.ndarray]]) -> None:
        """
        Index jobs with their embeddings in Weaviate.

        Args:
            jobs_data: List of tuples (job_dict, embedding_vector)
        """
        if not self.client:
            raise RuntimeError("Weaviate client not initialized")

        logger.info(f"Indexing {len(jobs_data)} jobs in Weaviate")

        jobs_collection = self.client.collections.get(self.jobs_class_name)

        # Prepare data for batch insert
        objects_to_insert = []
        for job, embedding in jobs_data:
            # Validate embedding dimension
            if len(embedding) != self.vector_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected "
                    f"{self.vector_dimension}, got {len(embedding)}"
                )
                continue

            weaviate_object = {
                "job_id": job["id"],
                "title": job["title"],
                "description": job["description"],
                "skills": job["skills"],
                "location_city": job["location"]["city"],
                "location_country": job["location"]["country"],
                "employment_type": job["employment_type"],
                "seniority": job["seniority"],
                "language": job["language"],
                "salary_min": job["salary_min"],
                "salary_max": job["salary_max"],
                "remote": job["remote"],
                "created_at": job["created_at"],
                "searchable_text": self._create_searchable_text(job, is_job=True),
            }

            objects_to_insert.append(
                DataObject(properties=weaviate_object, vector=embedding.tolist())
            )

        # Batch insert
        try:
            result = jobs_collection.data.insert_many(objects_to_insert)

            if result.has_errors:
                logger.error(f"Jobs indexing errors: {len(result.errors)}")
                for error in result.errors:
                    logger.error(f"Job indexing error: {error}")
            else:
                logger.info(f"Successfully indexed {len(result.all_responses)} jobs")

        except Exception as e:
            logger.error(f"Failed to index jobs: {e}")
            raise

    async def index_candidates(
        self, candidates_data: List[Tuple[Dict, np.ndarray]]
    ) -> None:
        """
        Index candidates with their embeddings in Weaviate.

        Args:
            candidates_data: List of tuples (candidate_dict, embedding_vector)
        """
        if not self.client:
            raise RuntimeError("Weaviate client not initialized")

        logger.info(f"Indexing {len(candidates_data)} candidates in Weaviate")

        candidates_collection = self.client.collections.get(self.candidates_class_name)

        # Prepare data for batch insert
        objects_to_insert = []
        for candidate, embedding in candidates_data:
            if len(embedding) != self.vector_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected "
                    f"{self.vector_dimension}, got {len(embedding)}"
                )
                continue

            weaviate_object = {
                "candidate_id": candidate["id"],
                "name": candidate["name"],
                "title": candidate["title"],
                "summary": candidate["summary"],
                "skills": candidate["skills"],
                "years_experience": candidate["years_experience"],
                "location_city": candidate["location"]["city"],
                "location_country": candidate["location"]["country"],
                "languages": candidate["languages"],
                "open_to_remote": candidate["open_to_remote"],
                "desired_salary": candidate["desired_salary"],
                "right_to_work_eu": candidate["right_to_work_eu"],
                "last_updated": candidate["last_updated"],
                "searchable_text": self._create_searchable_text(
                    candidate, is_job=False
                ),
            }

            objects_to_insert.append(
                DataObject(properties=weaviate_object, vector=embedding.tolist())
            )

        # Batch insert
        try:
            result = candidates_collection.data.insert_many(objects_to_insert)

            if result.has_errors:
                logger.error(f"Candidates indexing errors: {len(result.errors)}")
                for error in result.errors:
                    logger.error(f"Candidate indexing error: {error}")
            else:
                logger.info(
                    f"Successfully indexed {len(result.all_responses)} candidates"
                )

        except Exception as e:
            logger.error(f"Failed to index candidates: {e}")
            raise

    async def search_jobs(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, Dict, str]]:
        """
        Search jobs using Weaviate's native hybrid search.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of tuples (job_data, scores_dict, matched_snippet)
        """
        if not self.client:
            raise RuntimeError("Weaviate client not initialized")

        # Generate query vector
        query_vector = await embedding_service.encode_single(query)

        jobs_collection = self.client.collections.get(self.jobs_class_name)

        # Build where filter
        where_filter = self._build_where_filter(filters, is_job=True)

        try:
            logger.info(f"Performing hybrid search for jobs with query: '{query}'")

            response = jobs_collection.query.hybrid(
                query=query,
                vector=query_vector.tolist(),
                alpha=0.5,
                fusion_type=self.fusion_type,
                limit=top_k,
                filters=where_filter,
                return_properties=[
                    "job_id",
                    "title",
                    "description",
                    "skills",
                    "location_city",
                    "location_country",
                    "employment_type",
                    "seniority",
                    "language",
                    "salary_min",
                    "salary_max",
                    "remote",
                    "created_at",
                    "searchable_text",
                ],
                return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
            )

            results = []
            logger.info(f"Hybrid search returned {len(response.objects)} job objects")

            for obj in response.objects:
                try:
                    job_data = {
                        "id": obj.properties["job_id"],
                        "title": obj.properties["title"],
                        "description": obj.properties["description"],
                        "skills": obj.properties["skills"],
                        "location": {
                            "city": obj.properties["location_city"],
                            "country": obj.properties["location_country"],
                        },
                        "employment_type": obj.properties["employment_type"],
                        "seniority": obj.properties["seniority"],
                        "language": obj.properties["language"],
                        "salary_min": obj.properties["salary_min"],
                        "salary_max": obj.properties["salary_max"],
                        "remote": obj.properties["remote"],
                        "created_at": obj.properties["created_at"],
                    }

                    scores = self._extract_hybrid_scores(obj.metadata)

                    # Create snippet (use first part of description)
                    description = obj.properties["description"]
                    snippet = (
                        description[:200] + "..."
                        if len(description) > 200
                        else description
                    )

                    results.append((job_data, scores, snippet))

                except Exception as e:
                    logger.error(f"Error processing job result: {e}")
                    continue

            logger.info(f"Successfully processed {len(results)} job results")
            return results

        except Exception as e:
            logger.error(f"Hybrid search for jobs failed: {e}")
            raise

    async def search_candidates(
        self, query: str, top_k: int = 10, filters: Optional[Dict] = None
    ) -> List[Tuple[Dict, Dict, str]]:
        """
        Search candidates using hybrid search.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of tuples (candidate_data, scores_dict, matched_snippet)
        """
        if not self.client:
            raise RuntimeError("Weaviate client not initialized")

        # Generate query vector
        query_vector = await embedding_service.encode_single(query)

        candidates_collection = self.client.collections.get(self.candidates_class_name)

        # Build where filter
        where_filter = self._build_where_filter(filters, is_job=False)

        try:
            logger.info(
                f"Performing hybrid search for candidates with query: '{query}'"
            )

            response = candidates_collection.query.hybrid(
                query=query,
                vector=query_vector.tolist(),
                alpha=0.5,
                fusion_type=self.fusion_type,
                limit=top_k,
                filters=where_filter,
                return_properties=[
                    "candidate_id",
                    "name",
                    "title",
                    "summary",
                    "skills",
                    "years_experience",
                    "location_city",
                    "location_country",
                    "languages",
                    "open_to_remote",
                    "desired_salary",
                    "right_to_work_eu",
                    "last_updated",
                    "searchable_text",
                ],
                return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
            )

            results = []
            logger.info(
                f"Hybrid search returned {len(response.objects)} candidate objects"
            )

            for obj in response.objects:
                try:
                    candidate_data = {
                        "id": obj.properties["candidate_id"],
                        "name": obj.properties["name"],
                        "title": obj.properties["title"],
                        "summary": obj.properties["summary"],
                        "skills": obj.properties["skills"],
                        "years_experience": obj.properties["years_experience"],
                        "location": {
                            "city": obj.properties["location_city"],
                            "country": obj.properties["location_country"],
                        },
                        "languages": obj.properties["languages"],
                        "open_to_remote": obj.properties["open_to_remote"],
                        "desired_salary": obj.properties["desired_salary"],
                        "right_to_work_eu": obj.properties["right_to_work_eu"],
                        "last_updated": obj.properties["last_updated"],
                    }

                    scores = self._extract_hybrid_scores(obj.metadata)

                    # Create snippet (use summary)
                    summary = obj.properties["summary"]
                    snippet = summary[:200] + "..." if len(summary) > 200 else summary

                    results.append((candidate_data, scores, snippet))

                except Exception as e:
                    logger.error(f"Error processing candidate result: {e}")
                    continue

            logger.info(f"Successfully processed {len(results)} candidate results")
            return results

        except Exception as e:
            logger.error(f"Hybrid search for candidates failed: {e}")
            raise

    async def match_candidates_for_job(
        self, job_id: str, top_k: int = 10
    ) -> List[Tuple[Dict, Dict, str]]:
        """
        Find top candidates for a specific job by using the job details as search query.

        Args:
            job_id: ID of the job to find candidates for
            top_k: Number of candidates to return

        Returns:
            List of tuples (candidate_data, scores_dict, matched_snippet)
            Or empty list with error if job not found
        """
        try:
            logger.info(f"Matching candidates for job {job_id}")

            job_data = await self.get_job_by_id(job_id)
            if not job_data:
                logger.error(f"Job {job_id} not found")
                return []

            # Create a search query from job details
            search_query = self._create_job_search_query(job_data)
            logger.info(f"Using search query: '{search_query}'")

            # Create filters based on job requirements
            filters = self._create_candidate_filters_from_job(job_data)

            # Search candidates using the job-derived query and filters
            candidates = await self.search_candidates(search_query, top_k, filters)

            logger.info(f"Found {len(candidates)} matching candidates for job {job_id}")
            return candidates

        except Exception as e:
            logger.error(f"Error matching candidates for job {job_id}: {e}")
            return []

    def _create_job_search_query(self, job_data: Dict) -> str:
        """
        Create a search query from job data to find matching candidates.

        Args:
            job_data: Job dictionary with title, description, skills, etc.

        Returns:
            Search query string optimized for finding relevant candidates
        """
        # Combine key job elements into a search query
        query_parts = []

        if job_data.get("title"):
            query_parts.append(job_data["title"])

        if job_data.get("skills"):
            skills_text = " ".join(job_data["skills"][:5])
            query_parts.append(skills_text)

        if job_data.get("seniority"):
            query_parts.append(job_data["seniority"])

        if job_data.get("description"):
            description_words = job_data["description"].split()
            key_terms = [
                word
                for word in description_words[:50]
                if len(word) > 4
                and word.lower()
                not in [
                    "with",
                    "using",
                    "will",
                    "work",
                    "team",
                    "experience",
                    "required",
                    "responsible",
                    "development",
                    "applications",
                ]
            ]
            if key_terms:
                query_parts.append(" ".join(key_terms[:10]))

        search_query = " ".join(query_parts)
        return search_query[:500]

    def _create_candidate_filters_from_job(self, job_data: Dict) -> Optional[Dict]:
        """
        Create candidate search filters based on job requirements.

        Args:
            job_data: Job dictionary

        Returns:
            Filters dictionary for candidate search
        """
        filters = {}

        if job_data.get("language"):
            filters["language"] = job_data["language"]

        if job_data.get("remote") is not None:
            if not job_data["remote"]:
                # Location filtering for non-remote jobs
                # Only filter by country, not city, to be less restrictive
                if job_data.get("location") and job_data["location"].get("country"):
                    filters["location"] = job_data["location"]["country"]

        if job_data.get("skills"):
            filters["skills"] = job_data["skills"][:5]

        return filters if filters else None

    def _extract_hybrid_scores(self, metadata) -> Dict[str, float]:
        """
        Extract and format scores from Weaviate hybrid search metadata.

        Args:
            metadata: Weaviate hybrid search result metadata

        Returns:
            Dictionary with bm25, vector, and fused scores
        """
        fused_score = metadata.score if hasattr(metadata, "score") else 0.0
        scores = {"bm25": 0.0, "vector": 0.0, "fused": fused_score}

        if hasattr(metadata, "explain_score") and metadata.explain_score:
            explain = str(metadata.explain_score)

            if self.fusion_method_name == "ranked":
                bm25_match = re.search(r"Result Set keyword,bm25\).*?contributed\s*([\d\.]+)", explain, re.IGNORECASE | re.DOTALL)
                if bm25_match:
                    try:
                        scores["bm25"] = float(bm25_match.group(1))
                    except ValueError:
                        pass

                vector_match = re.search(r"Result Set vector,hybridVector\).*?contributed\s*([\d\.]+)", explain, re.IGNORECASE | re.DOTALL)
                if vector_match:
                    try:
                        scores["vector"] = float(vector_match.group(1))
                    except ValueError:
                        pass
            else:
                bm25_patterns = [
                    r"Result Set keyword,bm25.*?normalized score:\s*([\d\.]+)",
                    r"keyword,bm25.*?normalized score:\s*([\d\.]+)",
                    r"bm25.*?normalized score:\s*([\d\.]+)",
                ]

                for pattern in bm25_patterns:
                    match = re.search(pattern, explain, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            scores["bm25"] = float(match.group(1))
                            break
                        except ValueError:
                            continue

                vector_patterns = [
                    r"Result Set vector,hybridVector.*?normalized score:\s*([\d\.]+)",
                    r"vector,hybridVector.*?normalized score:\s*([\d\.]+)",
                    r"hybridVector.*?normalized score:\s*([\d\.]+)",
                ]

                for pattern in vector_patterns:
                    match = re.search(pattern, explain, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            scores["vector"] = float(match.group(1))
                            break
                        except ValueError:
                            continue

        scores = self._normalize_scores(scores)
        return scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize BM25 and vector scores to 0-1 range for better interpretation.e.

        Args:
            scores: Extracted scores dictionary

        Returns:
            Normalized scores dictionary
        """
        normalized = scores.copy()

        normalized["bm25"] = min(max(scores["bm25"], 0.0), 1.0)
        normalized["vector"] = min(max(scores["vector"], 0.0), 1.0)
        normalized["fused"] = min(max(scores["fused"], 0.0), 1.0)

        return normalized

    def _create_searchable_text(self, item: Dict, is_job: bool = True) -> str:
        """Create searchable text representation."""
        if is_job:
            skills = " ".join(item.get("skills", []))
            location = f"{item['location']['city']}, {item['location']['country']}"
            return (
                f"{item['title']} {item['description']} {skills} {location} "
                f"{item.get('employment_type', '')} {item.get('seniority', '')}"
            )
        else:
            skills = " ".join(item.get("skills", []))
            languages = " ".join(item.get("languages", []))
            location = f"{item['location']['city']}, {item['location']['country']}"
            return (
                f"{item['title']} {item.get('summary', '')} {skills} "
                f"{languages} {location}"
            )

    def _build_where_filter(
        self, filters: Optional[Dict], is_job: bool = True
    ) -> Optional[wvc.query.Filter]:
        """Build Weaviate where filter from search filters."""
        if not filters:
            return None

        filter_conditions = []

        if filters.get("location"):
            location_filter = filters["location"].lower()
            location_filter = wvc.query.Filter.by_property("location_city").like(
                f"*{location_filter}*"
            ) | wvc.query.Filter.by_property("location_country").like(
                f"*{location_filter}*"
            )
            filter_conditions.append(location_filter)

        if filters.get("remote") is not None:
            if is_job:
                remote_filter = wvc.query.Filter.by_property("remote").equal(
                    filters["remote"]
                )
            else:
                remote_filter = wvc.query.Filter.by_property("open_to_remote").equal(
                    filters["remote"]
                )
            filter_conditions.append(remote_filter)

        if filters.get("language"):
            if is_job:
                lang_filter = wvc.query.Filter.by_property("language").equal(
                    filters["language"].lower()
                )
            else:
                lang_filter = wvc.query.Filter.by_property("languages").contains_any(
                    [filters["language"].lower()]
                )
            filter_conditions.append(lang_filter)

        if filters.get("skills"):
            skills_filter = wvc.query.Filter.by_property("skills").contains_any(
                [skill.lower() for skill in filters["skills"]]
            )
            filter_conditions.append(skills_filter)

        if not filter_conditions:
            return None

        # Combine all conditions with AND
        result_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            result_filter = result_filter & condition

        return result_filter

    async def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """
        Retrieve a job by its ID from Weaviate.

        Args:
            job_id: ID of the job to retrieve

        Returns:
            Job dictionary if found, None otherwise
        """
        if not self.client:
            raise RuntimeError("Weaviate client not initialized")

        try:
            jobs_collection = self.client.collections.get(self.jobs_class_name)

            response = jobs_collection.query.fetch_objects(
                filters=wvc.query.Filter.by_property("job_id").equal(job_id), limit=1
            )

            if response.objects:
                obj = response.objects[0]
                return {
                    "id": obj.properties["job_id"],
                    "title": obj.properties["title"],
                    "description": obj.properties["description"],
                    "skills": obj.properties["skills"],
                    "location": {
                        "city": obj.properties["location_city"],
                        "country": obj.properties["location_country"],
                    },
                    "employment_type": obj.properties["employment_type"],
                    "seniority": obj.properties["seniority"],
                    "language": obj.properties["language"],
                    "salary_min": obj.properties["salary_min"],
                    "salary_max": obj.properties["salary_max"],
                    "remote": obj.properties["remote"],
                    "created_at": obj.properties["created_at"],
                }
            else:
                return None

        except Exception as e:
            logger.error(f"Error retrieving job {job_id}: {e}")
            return None

    def close(self) -> None:
        """Close the Weaviate client connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate hybrid search connection closed")


# Global hybrid search engine instance
hybrid_search_engine = WeaviateHybridSearchEngine(
    fusion_method=os.getenv("FUSION_METHOD", "relative_score")
)
