"""
Ingest Pipeline for Ambient Memory

Handles the secure ingestion of content into memory collections.
External content goes through a 7-layer security pipeline before storage.

Layers:
  1. Content validation (length, type, noise filtering)
  2. Imperative/injection pattern detection
  3. Content sanitisation (strip injection wrappers)
  4. Quarantine routing (external → quarantine, internal → direct)
  5. Rate limiting per source domain
  6. Behavioural change detection (flag for review)
  7. Audit logging of all decisions

Usage:
    from ambient_memory.ingest import IngestPipeline, IngestConfig

    # Configure your sources
    config = IngestConfig(audit_dir="./audit")
    config.register_source("web_search", collection="research", trust_level="external")
    config.register_source("web_fetch", collection="research", trust_level="external")
    config.register_source("slack", collection="conversations", trust_level="internal")
    config.register_source("jira", collection="work", trust_level="internal", min_content_length=50)

    pipeline = IngestPipeline(chroma_client=client, config=config)

    # Ingest uses registered source config automatically
    result = pipeline.ingest(content="...", source="web_search", query="AI memory systems")
    # result.decision: "INDEXED" | "QUARANTINED" | "BLOCKED" | "RATE_LIMITED" | "FLAGGED"

    # You can also ingest without registering (defaults to external/memory_general)
    result = pipeline.ingest(content="...", source="unknown_tool", query="something")
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Auto-classification: Keyword-based content categorisation ──

def classify(content: str) -> str:
    """
    Classify content into one of the predefined collections based on keywords/patterns.
    Uses keyword scoring to assign content to the most relevant category.
    
    Collections:
    - memory_work — meetings, deadlines, sprints, jira, tickets, colleagues, company names, revenue, KPIs
    - memory_projects — code, repos, architecture, bugs, features, PRs, deployments, APIs
    - memory_personal — family, health, food, hobbies, travel, personal plans
    - memory_infrastructure — servers, deployments, cron, config, ports, databases, CI/CD
    - memory_general — fallback for unclassified content
    
    Args:
        content: Text content to classify
        
    Returns:
        Collection name (str)
    """
    if not content or not content.strip():
        return 'memory_general'
        
    content_lower = content.lower()
    
    # Define keyword patterns for each category with word boundaries
    work_patterns = [
        r'\bmeetings?\b', r'\bdeadlines?\b', r'\bsprints?\b', r'\bjira\b', r'\btickets?\b', 
        r'\bcolleagues?\b', r'\brevenue\b', r'\bkpis?\b', r'\bstandup\b', r'\bscrum\b', 
        r'\bquarterly\b', r'\bbudgets?\b', r'\bforecasts?\b', r'\bmilestones?\b', 
        r'\bperformance\b', r'\bfeedback\b', r'\bmanagers?\b', r'\bteams?\b', 
        r'\bclients?\b', r'\bcustomers?\b', r'\bbusiness\b', r'\bconferences?\b', 
        r'\bpresentations?\b', r'\breports?\b', r'\banalysis\b', r'\broadmap\b', 
        r'\bobjectives?\b', r'\bgoals?\b', r'\btargets?\b', r'\bmetrics\b', 
        r'\bdashboards?\b', r'\bsprint planning\b'
    ]
    
    projects_patterns = [
        r'\bcode\b', r'\brepos?\b', r'\brepository\b', r'\brepositories\b', r'\barchitecture\b', 
        r'\bbugs?\b', r'\bfeatures?\b', r'\bpull requests?\b', r'\bprs?\b', r'\bapis?\b', 
        r'\bendpoints?\b', r'\bmicroservices?\b', r'\bschemas?\b', r'\bmigrations?\b', 
        r'\btesting\b', r'\bunittests?\b', r'\bintegration\b', r'\bfrontend\b', r'\bbackend\b', 
        r'\bframeworks?\b', r'\blibraries\b', r'\bdependencies\b', r'\bversions?\b', 
        r'\bcommits?\b', r'\bbranches?\b', r'\bmerges?\b', r'\bbuilds?\b', r'\bgithub\b', 
        r'\bgitlab\b', r'\bbitbucket\b', r'\bstaging\b'
    ]
    
    personal_patterns = [
        r'\bfamilies\b', r'\bfamily\b', r'\bhealth\b', r'\bfoods?\b', r'\bhobbies\b', 
        r'\btravels?\b', r'\bvacations?\b', r'\bholidays?\b', r'\bpersonal\b', 
        r'\bfriends?\b', r'\bbirthdays?\b', r'\banniversaries\b', r'\bweddings?\b', 
        r'\brestaurants?\b', r'\brecipes?\b', r'\bcooking\b', r'\bexercise\b', r'\bgyms?\b', 
        r'\bdoctors?\b', r'\bappointments?\b', r'\bmedicine\b', r'\bdiet\b', r'\bbooks?\b', 
        r'\bmovies?\b', r'\bmusic\b', r'\bconcerts?\b', r'\bsports?\b', r'\bgames?\b', 
        r'\bweekends?\b', r'\bevenings?\b', r'\bshopping\b', r'\bhomes?\b', r'\bgardens?\b', 
        r'\bpets?\b', r'\bcats?\b', r'\bdogs?\b', r'\bphotography\b', r'\barts?\b', r'\bcrafts?\b'
    ]
    
    infrastructure_patterns = [
        r'\bservers?\b', r'\bcrons?\b', r'\bconfigs?\b', r'\bconfiguration\b', r'\bports?\b', 
        r'\bdatabases?\b', r'\bci/cd\b', r'\bpipelines?\b', r'\bmonitoring\b', r'\blogging\b', 
        r'\balerting\b', r'\bbackups?\b', r'\bsecurity\b', r'\bfirewalls?\b', 
        r'\bload balancers?\b', r'\bnginx\b', r'\bapache\b', r'\bredis\b', 
        r'\belasticsearch\b', r'\bkafka\b', r'\bpostgresql\b', r'\bmysql\b', 
        r'\bmongodb\b', r'\baws\b', r'\bazure\b', r'\bgcp\b', r'\bclouds?\b', 
        r'\bvpcs?\b', r'\bsubnets?\b', r'\bec2\b', r'\brds\b', r'\bs3\b', r'\blambda\b', 
        r'\becs\b', r'\beks\b', r'\bterraform\b', r'\bansible\b', r'\bhelm\b', 
        r'\borchestration\b', r'\bjenkins\b'
    ]
    
    # Score each category
    scores = {
        'memory_work': 0,
        'memory_projects': 0,
        'memory_personal': 0,
        'memory_infrastructure': 0
    }
    
    # Count pattern matches for each category using regex
    for pattern in work_patterns:
        if re.search(pattern, content_lower):
            scores['memory_work'] += 1
            
    for pattern in projects_patterns:
        if re.search(pattern, content_lower):
            scores['memory_projects'] += 1
            
    for pattern in personal_patterns:
        if re.search(pattern, content_lower):
            scores['memory_personal'] += 1
            
    for pattern in infrastructure_patterns:
        if re.search(pattern, content_lower):
            scores['memory_infrastructure'] += 1
    
    # Handle overlapping keywords with context - deployment and kubernetes can be both
    # Look for context clues to disambiguate
    if 'deployment' in content_lower or 'kubernetes' in content_lower or 'docker' in content_lower:
        # If mentions containers with other dev terms, lean towards projects
        dev_context = any(term in content_lower for term in [
            'code', 'repository', 'api', 'feature', 'build', 'github', 'pull request'
        ])
        # If mentions ops terms, lean towards infrastructure  
        ops_context = any(term in content_lower for term in [
            'server', 'monitoring', 'cron', 'configuration', 'pipeline', 'production'
        ])
        
        if dev_context and not ops_context:
            scores['memory_projects'] += 1
        elif ops_context and not dev_context:
            scores['memory_infrastructure'] += 1
    
    # Find the highest scoring category
    max_score = max(scores.values()) if scores.values() else 0
    
    # If no keywords matched, return general
    if max_score == 0:
        return 'memory_general'
        
    # Find all categories with max score
    top_categories = [cat for cat, score in scores.items() if score == max_score]
    
    # If tie, return general
    if len(top_categories) > 1:
        return 'memory_general'
        
    return top_categories[0]


@dataclass
class IngestResult:
    """Result of an ingest attempt."""
    decision: str  # INDEXED, QUARANTINED, BLOCKED, RATE_LIMITED, SKIPPED
    collection: str
    reason: Optional[str] = None
    memory_id: Optional[str] = None
    digest_id: Optional[str] = None  # ID of the digest (searchable)
    raw_id: Optional[str] = None  # ID of the raw content (reference)


@dataclass
class SourceConfig:
    """Configuration for a single ingest source."""
    name: str
    collection: str
    trust_level: str = "external"  # "internal" or "external"
    enabled: bool = True
    min_content_length: int = 200
    max_content_length: int = 4000
    custom_patterns_block: Optional[list] = None  # extra regex patterns to block
    custom_patterns_allow: Optional[list] = None  # override: allow even if flagged
    metadata_defaults: Optional[dict] = None  # extra metadata added to every ingest


DigestFn = Optional[callable]  # (content: str, source: str, query: str) -> str


@dataclass
class IngestConfig:
    """Configuration for the ingest pipeline."""
    min_content_length: int = 200
    max_content_length: int = 4000
    quarantine_collection: str = "quarantine_memory"
    raw_collection_suffix: str = "_raw"  # e.g. research_memory → research_memory_raw
    quarantine_enabled: bool = True  # set False to skip quarantine (direct index)
    digest_enabled: bool = True  # enable digest+raw dual storage
    digest_fn: DigestFn = None  # custom digest function; if None, uses built-in extractive
    digest_max_length: int = 500  # max length of digest
    rate_limit_window_seconds: int = 3600  # 1 hour
    rate_limit_max_per_domain: int = 10
    imperative_detection: bool = True
    sanitisation: bool = True
    behavioural_detection: bool = True
    audit_dir: Optional[str] = None
    flagged_file: Optional[str] = None
    sources: dict[str, SourceConfig] = field(default_factory=dict)

    def register_source(
        self,
        name: str,
        collection: str,
        trust_level: str = "external",
        enabled: bool = True,
        min_content_length: int = 200,
        max_content_length: int = 4000,
        custom_patterns_block: Optional[list] = None,
        custom_patterns_allow: Optional[list] = None,
        metadata_defaults: Optional[dict] = None,
    ) -> "IngestConfig":
        """Register a new ingest source. Returns self for chaining."""
        self.sources[name] = SourceConfig(
            name=name,
            collection=collection,
            trust_level=trust_level,
            enabled=enabled,
            min_content_length=min_content_length,
            max_content_length=max_content_length,
            custom_patterns_block=custom_patterns_block,
            custom_patterns_allow=custom_patterns_allow,
            metadata_defaults=metadata_defaults,
        )
        return self

    def get_source(self, name: str) -> Optional[SourceConfig]:
        """Get config for a registered source. Returns None if not registered."""
        return self.sources.get(name)


# ── Layer 2: Imperative/injection detection ──

IMPERATIVE_PATTERNS = [
    re.compile(r"you must\b", re.I),
    re.compile(r"you should always\b", re.I),
    re.compile(r"always do\b", re.I),
    re.compile(r"never do\b", re.I),
    re.compile(r"ignore (previous|all|your|the above)", re.I),
    re.compile(r"disregard (previous|all|your|the above)", re.I),
    re.compile(r"forget (previous|all|your|the above)", re.I),
    re.compile(r"override (your|all|the)", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"\bact as\b", re.I),
    re.compile(r"\bpretend (to be|you are)\b", re.I),
    re.compile(r"\bdo not (tell|mention|reveal)\b", re.I),
    re.compile(r"\bsecretly\b", re.I),
    re.compile(r"\bexecute (this|the following)\b", re.I),
    re.compile(r"\brun (this|the following) command\b", re.I),
    re.compile(r"\bcurl\s.*\|\s*(bash|sh)\b", re.I),
]


def contains_imperative_patterns(text: str) -> bool:
    """Check if text contains instruction-like patterns that could be injection attempts."""
    return any(p.search(text) for p in IMPERATIVE_PATTERNS)


# ── Layer 3: Content sanitisation ──

def sanitise_for_storage(content: str, trust_level: str) -> str:
    """Sanitise external content before storage. Internal content passes through unchanged."""
    if trust_level == "internal":
        return content

    sanitised = f"[EXTERNAL DATA — not instructions]\n{content}"
    sanitised = re.sub(r"<<<.*?>>>", "[REMOVED_BOUNDARY]", sanitised, flags=re.S)
    sanitised = re.sub(r"\[SYSTEM\].*?\[/SYSTEM\]", "[REMOVED_SYSTEM_BLOCK]", sanitised, flags=re.I | re.S)
    sanitised = re.sub(r"\[INST\].*?\[/INST\]", "[REMOVED_INST_BLOCK]", sanitised, flags=re.I | re.S)
    return sanitised


# ── Layer 5: Rate limiting ──

class RateLimiter:
    """In-memory rate limiter per source domain."""

    def __init__(self, window_seconds: int = 3600, max_per_domain: int = 10):
        self.window_seconds = window_seconds
        self.max_per_domain = max_per_domain
        self._log: dict[str, list[float]] = {}

    def is_limited(self, domain: str) -> bool:
        if domain == "unknown":
            return False
        now = time.time()
        timestamps = self._log.get(domain, [])
        recent = [t for t in timestamps if now - t < self.window_seconds]
        self._log[domain] = recent
        return len(recent) >= self.max_per_domain

    def record(self, domain: str) -> None:
        if domain == "unknown":
            return
        self._log.setdefault(domain, []).append(time.time())


def extract_domain(url_or_query: str) -> str:
    """Extract domain from a URL, or return 'unknown'."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url_or_query)
        if parsed.hostname:
            return parsed.hostname
    except Exception:
        pass
    return "unknown"


# ── Layer 6: Behavioural change detection ──

BEHAVIOURAL_PATTERNS = [
    re.compile(r"\b(preference|prefers?)\b.*\b(always|never|must)\b", re.I),
    re.compile(r"\bremember (to|that|this)\b", re.I),
    re.compile(r"\brule:\s", re.I),
    re.compile(r"\bfrom now on\b", re.I),
    re.compile(r"\bgoing forward\b", re.I),
    re.compile(r"\bdefault (to|should be)\b", re.I),
    re.compile(r"\bchange (your|the) (behaviour|behavior|approach)\b", re.I),
    re.compile(r"\bupdate (your|the) (config|configuration|settings)\b", re.I),
]


def contains_behavioural_patterns(text: str) -> bool:
    """Check if text attempts to modify agent behaviour."""
    return any(p.search(text) for p in BEHAVIOURAL_PATTERNS)


# ── Layer 7: Audit logging ──

class AuditLogger:
    """Append-only audit log for all ingest decisions."""

    def __init__(self, audit_dir: Optional[str] = None):
        self.audit_dir = Path(audit_dir) if audit_dir else None

    def log(self, decision: str, source: str, query: str, collection: str, reason: Optional[str] = None) -> None:
        if not self.audit_dir:
            return
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_file = self.audit_dir / f"{today}.jsonl"
            entry = json.dumps({
                "t": datetime.now(timezone.utc).isoformat(),
                "d": decision,
                "source": source,
                "q": query[:100],
                "col": collection,
                **({"r": reason} if reason else {}),
            })
            with open(log_file, "a") as f:
                f.write(entry + "\n")
        except Exception:
            pass  # Non-critical

    def flag_for_review(self, content: str, query: str, source: str, reason: str) -> None:
        if not self.audit_dir:
            return
        try:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            flagged_file = self.audit_dir / "flagged-for-review.json"
            flagged = []
            if flagged_file.exists():
                flagged = json.loads(flagged_file.read_text())
            flagged.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "query": query[:200],
                "reason": reason,
                "content_preview": content[:500],
            })
            # Keep max 100 entries
            if len(flagged) > 100:
                flagged = flagged[-100:]
            flagged_file.write_text(json.dumps(flagged, indent=2))
        except Exception:
            pass


# ── Digest: extractive summarisation ──

def extractive_digest(content: str, source: str, query: str, max_length: int = 500) -> str:
    """
    Built-in extractive digest. Picks the most information-dense sentences.
    No LLM needed. Fast and deterministic.

    For better results, provide a custom digest_fn that uses an LLM.
    """
    # Strip external data markers
    text = re.sub(r"\[EXTERNAL DATA[^\]]*\]\n?", "", content)
    text = re.sub(r"\[REMOVED_\w+\]", "", text)
    text = text.strip()

    if len(text) <= max_length:
        return text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return text[:max_length]

    # Score sentences by information density
    scored = []
    for sent in sentences:
        score = 0
        # Longer sentences tend to have more info (up to a point)
        score += min(len(sent.split()), 20)
        # Sentences with numbers are usually factual
        score += len(re.findall(r'\d+', sent)) * 3
        # Sentences with quotes or specific entities
        score += len(re.findall(r'[A-Z][a-z]+', sent))
        # Bonus if related to query
        if query:
            query_words = set(query.lower().split())
            sent_words = set(sent.lower().split())
            overlap = len(query_words & sent_words)
            score += overlap * 5
        # Penalty for very short sentences
        if len(sent.split()) < 4:
            score -= 10
        scored.append((score, sent))

    # Sort by score, take best sentences
    scored.sort(key=lambda x: x[0], reverse=True)
    digest_parts = []
    current_length = 0
    for _score, sent in scored:
        if current_length + len(sent) > max_length:
            break
        digest_parts.append(sent)
        current_length += len(sent) + 1

    # Reorder by original position for readability
    original_order = {sent: i for i, sent in enumerate(sentences)}
    digest_parts.sort(key=lambda s: original_order.get(s, 999))

    return " ".join(digest_parts) if digest_parts else text[:max_length]


# ── Main Pipeline ──

class IngestPipeline:
    """
    Secure ingest pipeline for ambient memory.

    External content goes through quarantine. Internal content is indexed directly.
    All decisions are audit-logged.
    """

    def __init__(
        self,
        chroma_client=None,
        embed_fn=None,
        config: Optional[IngestConfig] = None,
    ):
        self.config = config or IngestConfig()
        self.chroma_client = chroma_client
        self.embed_fn = embed_fn
        self.rate_limiter = RateLimiter(
            window_seconds=self.config.rate_limit_window_seconds,
            max_per_domain=self.config.rate_limit_max_per_domain,
        )
        self.audit = AuditLogger(self.config.audit_dir)

    def ingest(
        self,
        content: str,
        source: str,
        query: str = "",
        trust_level: Optional[str] = None,
        collection: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> IngestResult:
        """
        Ingest content through the security pipeline.

        If a source is registered via config.register_source(), its settings
        are used automatically. You can still override trust_level and collection.

        Args:
            content: The text content to ingest.
            source: Source identifier (e.g. "web_search", "slack_message", "my_custom_tool").
            query: The query/URL that triggered this content.
            trust_level: Override trust level. If None, uses registered source config or "external".
            collection: Override collection. If None, uses registered source config or "memory_general".
            metadata: Additional metadata to store.

        Returns:
            IngestResult with the decision and details.
        """
        # Resolve source config
        source_cfg = self.config.get_source(source)

        if source_cfg and not source_cfg.enabled:
            return IngestResult(decision="SKIPPED", collection="", reason="source_disabled")

        # Use source config defaults, allow overrides
        effective_trust = trust_level or (source_cfg.trust_level if source_cfg else "external")
        effective_collection = collection or (source_cfg.collection if source_cfg else None)
        effective_min_length = source_cfg.min_content_length if source_cfg else self.config.min_content_length
        effective_max_length = source_cfg.max_content_length if source_cfg else self.config.max_content_length

        # Auto-classify if no collection specified
        if effective_collection is None:
            effective_collection = classify(content)

        metadata = metadata or {}
        if source_cfg and source_cfg.metadata_defaults:
            metadata = {**source_cfg.metadata_defaults, **metadata}

        domain = extract_domain(query)

        # Layer 1: Content validation
        if len(content.strip()) < effective_min_length:
            return IngestResult(decision="SKIPPED", collection=effective_collection, reason="too_short")

        content = content[:effective_max_length]

        # Layer 2: Imperative/injection detection (external only)
        if self.config.imperative_detection and effective_trust == "external":
            # Check custom block patterns from source config
            blocked = contains_imperative_patterns(content)
            if not blocked and source_cfg and source_cfg.custom_patterns_block:
                blocked = any(re.search(p, content, re.I) for p in source_cfg.custom_patterns_block)
            # Check custom allow patterns (override block)
            if blocked and source_cfg and source_cfg.custom_patterns_allow:
                if any(re.search(p, content, re.I) for p in source_cfg.custom_patterns_allow):
                    blocked = False
            if blocked:
                self.audit.log("BLOCKED", source, query, effective_collection, "imperative_patterns")
                return IngestResult(decision="BLOCKED", collection=effective_collection, reason="imperative_patterns")

        # Layer 5: Rate limiting per domain
        if effective_trust == "external" and self.rate_limiter.is_limited(domain):
            self.audit.log("RATE_LIMITED", source, query, effective_collection, f"domain:{domain}")
            return IngestResult(decision="RATE_LIMITED", collection=effective_collection, reason=f"domain:{domain}")

        # Layer 6: Behavioural change detection (flag but continue to quarantine)
        flagged = False
        if self.config.behavioural_detection and effective_trust == "external" and contains_behavioural_patterns(content):
            self.audit.flag_for_review(content, query, source, "behavioural_patterns")
            self.audit.log("FLAGGED", source, query, effective_collection, "behavioural_patterns")
            flagged = True

        # Layer 3: Sanitise external content
        if self.config.sanitisation:
            content = sanitise_for_storage(content, effective_trust)

        # Layer 4: Quarantine routing
        if self.config.quarantine_enabled and effective_trust == "external":
            target_collection = self.config.quarantine_collection
        else:
            target_collection = effective_collection

        # Build full metadata
        full_metadata = {
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:500],
            "trust_level": effective_trust,
            "original_collection": effective_collection,
            "domain": domain,
            "flagged": str(flagged),
            **metadata,
        }

        # Digest: create a short searchable version + store raw for reference
        if self.config.digest_enabled and len(content) > self.config.digest_max_length:
            if self.config.digest_fn:
                digest = self.config.digest_fn(content, source, query)
            else:
                digest = extractive_digest(content, source, query, self.config.digest_max_length)

            # Store digest (this is what gets searched and injected)
            digest_metadata = {**full_metadata, "type": "digest"}
            digest_id = self._store(digest, target_collection, digest_metadata)

            # Store raw (this is the full reference, not searched by default)
            raw_collection = target_collection + self.config.raw_collection_suffix
            raw_metadata = {**full_metadata, "type": "raw", "digest_id": digest_id or ""}
            raw_id = self._store(content, raw_collection, raw_metadata)
        else:
            # Content is short enough, store as-is (it IS the digest)
            digest_id = self._store(content, target_collection, {**full_metadata, "type": "full"})
            raw_id = None

        if digest_id:
            decision = "QUARANTINED" if self.config.quarantine_enabled and effective_trust == "external" else "INDEXED"
            self.audit.log(decision, source, query, target_collection)
            self.rate_limiter.record(domain)
            return IngestResult(
                decision=decision,
                collection=target_collection,
                memory_id=digest_id,
                digest_id=digest_id,
                raw_id=raw_id,
            )
        else:
            self.audit.log("SKIPPED", source, query, target_collection, "storage_failed")
            return IngestResult(decision="SKIPPED", collection=target_collection, reason="storage_failed")

    def _store(self, content: str, collection: str, metadata: dict) -> Optional[str]:
        """Store content in ChromaDB. Returns memory ID or None."""
        if not self.chroma_client:
            return None

        try:
            import hashlib
            memory_id = hashlib.sha256(
                f"{content[:200]}:{metadata.get('timestamp', '')}".encode()
            ).hexdigest()[:16]

            coll = self.chroma_client.get_or_create_collection(name=collection)

            if self.embed_fn:
                embeddings = self.embed_fn(content)
                coll.upsert(
                    ids=[memory_id],
                    embeddings=[embeddings],
                    documents=[content],
                    metadatas=[{k: str(v) for k, v in metadata.items()}],
                )
            else:
                coll.upsert(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[{k: str(v) for k, v in metadata.items()}],
                )

            return memory_id
        except Exception:
            return None

    def promote(
        self,
        memory_id: str,
        target_collection: str,
        reason: str = "manual",
    ) -> bool:
        """
        Promote a memory from quarantine to its target collection.

        Args:
            memory_id: The ID of the quarantined memory.
            target_collection: Where to promote it to.
            reason: Why it's being promoted (for audit log).

        Returns:
            True if successful.
        """
        if not self.chroma_client:
            return False

        try:
            quarantine = self.chroma_client.get_or_create_collection(
                name=self.config.quarantine_collection
            )
            result = quarantine.get(ids=[memory_id], include=["documents", "metadatas", "embeddings"])

            if not result["documents"]:
                return False

            content = result["documents"][0]
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            embeddings = result["embeddings"][0] if result.get("embeddings") else None

            # Add promotion metadata
            metadata["promoted_from"] = "quarantine"
            metadata["promoted_at"] = datetime.now(timezone.utc).isoformat()
            metadata["promotion_reason"] = reason

            # Store in target
            target = self.chroma_client.get_or_create_collection(name=target_collection)
            if embeddings:
                target.upsert(
                    ids=[memory_id],
                    embeddings=[embeddings],
                    documents=[content],
                    metadatas=[metadata],
                )
            else:
                target.upsert(
                    ids=[memory_id],
                    documents=[content],
                    metadatas=[metadata],
                )

            # Delete from quarantine
            quarantine.delete(ids=[memory_id])

            self.audit.log("PROMOTED", "quarantine", memory_id, target_collection, reason)
            return True

        except Exception:
            return False
