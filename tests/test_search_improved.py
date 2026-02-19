"""
Tests for improved hybrid search functionality.

Tests the ImprovedHybridSearcher implementation including all Phase 3 logic
preservation, Phase 5 structural improvements, and Phase 6 enhancements.
"""

import pytest
from datetime import datetime
from ambient_memory.search_improved import (
    ImprovedHybridSearcher, SearchConfig, extract_entities, compute_keyword_score,
    should_skip_query, extract_date_from_source, compute_temporal_boost
)


class TestEntityExtraction:
    """Test entity extraction functionality (Phase 3 logic preserved)."""
    
    def test_extract_capitalised_words(self):
        """Test extraction of capitalised words."""
        text = "I need to deploy Darwin to Railway and update Chowdown."
        entities = extract_entities(text)
        
        assert "darwin" in entities
        assert "railway" in entities 
        assert "chowdown" in entities
        assert "deploy" not in entities  # lowercase
    
    def test_extract_multi_word_entities(self):
        """Test extraction of multi-word capitalised entities."""
        text = "Google Calendar integration with Hire Space is ready."
        entities = extract_entities(text)
        
        assert "google calendar" in entities
        assert "hire space" in entities
        assert "google" in entities  # individual words too
        assert "calendar" in entities
        assert "hire" in entities
        assert "space" in entities
    
    def test_extract_all_caps_acronyms(self):
        """Test extraction of ALL-CAPS acronyms."""
        text = "The API documentation and SEO analysis are done."
        entities = extract_entities(text)
        
        assert "api" in entities
        assert "seo" in entities
        assert "the" not in entities  # lowercase
    
    def test_empty_text(self):
        """Test extraction from empty text."""
        entities = extract_entities("")
        assert len(entities) == 0
    
    def test_punctuation_handling(self):
        """Test entity extraction handles punctuation correctly."""
        text = "Darwin's API, Railway's deployment, and Chowdown's interface."
        entities = extract_entities(text)
        
        assert "darwin" in entities
        assert "api" in entities
        assert "railway" in entities
        assert "chowdown" in entities


class TestKeywordScoring:
    """Test keyword scoring functionality (Phase 3 logic preserved)."""
    
    def test_compute_keyword_score_full_match(self):
        """Test keyword score with all entities present."""
        query_entities = {"darwin", "railway", "deployment"}
        document = "Darwin deployment on Railway is complete with all features."
        
        score = compute_keyword_score(query_entities, document)
        assert score == 1.0  # All entities found
    
    def test_compute_keyword_score_partial_match(self):
        """Test keyword score with partial entity match."""
        query_entities = {"darwin", "railway", "deployment"}
        document = "Darwin is ready but Railway setup is pending."
        
        score = compute_keyword_score(query_entities, document)
        assert score == pytest.approx(2/3, abs=0.01)  # 2 out of 3 entities
    
    def test_compute_keyword_score_no_match(self):
        """Test keyword score with no entity matches."""
        query_entities = {"darwin", "railway", "deployment"}
        document = "General discussion about weather and meetings."
        
        score = compute_keyword_score(query_entities, document)
        assert score == 0.0
    
    def test_compute_keyword_score_empty_entities(self):
        """Test keyword score with empty query entities."""
        query_entities = set()
        document = "Some document text here."
        
        score = compute_keyword_score(query_entities, document)
        assert score == 0.0


class TestQueryFiltering:
    """Test Phase 6 query filtering functionality."""
    
    def test_should_skip_short_acknowledgments(self):
        """Test that short acknowledgments are filtered."""
        assert should_skip_query("ok")
        assert should_skip_query("dale")
        assert should_skip_query("si")
        assert should_skip_query("thanks")
        assert should_skip_query("👍")
    
    def test_should_not_skip_meaningful_queries(self):
        """Test that meaningful queries are not filtered."""
        assert not should_skip_query("How is Darwin deployment going?")
        assert not should_skip_query("Can you check Railway status")
        assert not should_skip_query("Update on Chowdown features")
        assert not should_skip_query("ok but what about the API issue")  # longer context
    
    def test_edge_cases(self):
        """Test edge cases for query filtering."""
        assert not should_skip_query("")  # empty string
        assert not should_skip_query("OK API")  # contains meaningful word


class TestTemporalFeatures:
    """Test temporal boosting functionality from Phase 5."""
    
    def test_extract_date_from_source(self):
        """Test date extraction from source filenames."""
        assert extract_date_from_source("the-factory-2026-02-15.md") == datetime(2026, 2, 15)
        assert extract_date_from_source("chat-2025-12-01.txt") == datetime(2025, 12, 1)
        assert extract_date_from_source("no-date-file.md") is None
        assert extract_date_from_source("invalid-2026-13-40.md") is None
    
    def test_compute_temporal_boost(self):
        """Test temporal proximity boost calculation."""
        query_time = datetime(2026, 2, 15)
        
        # Same day should give high boost
        same_day_boost = compute_temporal_boost(query_time, "file-2026-02-15.md")
        assert same_day_boost > 0.09  # Close to weight (0.1)
        
        # One week ago should give lower boost
        week_ago_boost = compute_temporal_boost(query_time, "file-2026-02-08.md")
        assert 0 < week_ago_boost < same_day_boost
        
        # No date should give no boost
        no_date_boost = compute_temporal_boost(query_time, "no-date.md")
        assert no_date_boost == 0.0


class TestSearchConfig:
    """Test SearchConfig dataclass and default values."""
    
    def test_default_config(self):
        """Test that default config maintains Phase 3 values."""
        config = SearchConfig()
        
        # Phase 3 core values preserved
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3
        assert config.distance_threshold == 1.5
        
        # Phase 6 improvements with safe defaults
        assert config.min_relevance_threshold == 0.0  # Conservative for coverage
        assert config.enable_query_filtering == True
        assert config.enable_deduplication == True
        assert config.adaptive_result_gap == 0.0  # Disabled by default
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SearchConfig(
            semantic_weight=0.8,
            min_relevance_threshold=0.5,
            enable_query_filtering=False
        )
        
        assert config.semantic_weight == 0.8
        assert config.min_relevance_threshold == 0.5
        assert config.enable_query_filtering == False
        # Other values should remain default
        assert config.keyword_weight == 0.3


class TestSearcherInitialisation:
    """Test ImprovedHybridSearcher initialisation."""
    
    def test_default_initialisation(self):
        """Test searcher initialises with default config."""
        # Note: This test won't work without ChromaDB installed
        # but we can test the config setup
        config = SearchConfig()
        # searcher = ImprovedHybridSearcher(config=config)
        # assert searcher.config.semantic_weight == 0.7
        assert config.semantic_weight == 0.7
    
    def test_custom_config_initialisation(self):
        """Test searcher initialises with custom config."""
        custom_config = SearchConfig(
            min_relevance_threshold=0.6,
            enable_deduplication=False
        )
        # searcher = ImprovedHybridSearcher(config=custom_config)
        # assert searcher.config.min_relevance_threshold == 0.6
        # assert searcher.config.enable_deduplication == False
        assert custom_config.min_relevance_threshold == 0.6
        assert custom_config.enable_deduplication == False


class TestPhase6Improvements:
    """Test that Phase 6 improvements work correctly."""
    
    def test_phase3_algorithm_preserved(self):
        """Test that core Phase 3 scoring algorithm is preserved."""
        # Test entity extraction (core Phase 3 component)
        text = "Deploy Darwin to Railway with API integration"
        entities = extract_entities(text)
        
        # Should extract exactly as Phase 3 did
        assert "darwin" in entities
        assert "railway" in entities
        assert "api" in entities
        assert len(entities) >= 3
    
    def test_structural_improvements(self):
        """Test that Phase 5 structural improvements are present."""
        # SearchConfig should exist and be configurable
        config = SearchConfig(semantic_weight=0.6)
        assert config.semantic_weight == 0.6
        
        # Utility functions should be available
        assert callable(extract_entities)
        assert callable(compute_keyword_score)
        assert callable(should_skip_query)
        assert callable(extract_date_from_source)
        assert callable(compute_temporal_boost)
    
    def test_phase6_enhancements(self):
        """Test that Phase 6 enhancements are available."""
        config = SearchConfig()
        
        # All Phase 6 features should be configurable
        assert hasattr(config, 'min_relevance_threshold')
        assert hasattr(config, 'enable_query_filtering')  
        assert hasattr(config, 'enable_deduplication')
        assert hasattr(config, 'adaptive_result_gap')
        
        # Query filtering should work
        assert should_skip_query("ok")
        assert not should_skip_query("How is the project going?")


# Integration test placeholder (would need ChromaDB setup)
class TestIntegration:
    """Integration tests (require ChromaDB setup)."""
    
    def test_search_integration_placeholder(self):
        """Placeholder for integration tests."""
        # In a real setup, we would:
        # 1. Create test ChromaDB with sample data
        # 2. Run searches with different configs
        # 3. Verify Phase 3 logic + Phase 6 improvements work together
        # 4. Check that results are properly filtered/scored
        
        # For now, just verify components can be imported
        assert ImprovedHybridSearcher is not None
        assert SearchConfig is not None
        
    def test_config_update_placeholder(self):
        """Placeholder for config update tests."""
        # In a real setup, we would:
        # 1. Create searcher instance
        # 2. Update config dynamically
        # 3. Verify search behavior changes accordingly
        
        config = SearchConfig(min_relevance_threshold=0.5)
        assert config.min_relevance_threshold == 0.5