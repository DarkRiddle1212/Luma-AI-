"""
tests/property/test_storage_properties.py

Property-based tests for the Luma Persistence & Storage Layer.

Uses Hypothesis to verify universal correctness properties defined in the
design document.

Feature: luma-persistence-storage
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from luma.storage.database import DatabaseManager
from luma.storage.migrations import MigrationRunner, _MIGRATIONS


# ---------------------------------------------------------------------------
# Property 12: Migration monotonicity
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 12: Migration monotonicity
@given(starting_version=st.integers(min_value=0, max_value=len(_MIGRATIONS) - 1))
@settings(max_examples=100)
def test_migration_monotonicity(starting_version: int) -> None:
    """
    Property: For any valid starting schema version (0 through N−1), applying
    all pending migrations via MigrationRunner.run_pending() SHALL result in a
    schema_version equal to the total count of defined migrations, and no
    migration SHALL be applied more than once.

    Also verifies idempotency: calling run_pending() a second time does not
    change the version.

    **Validates: Requirements 9.2, 9.6, 9.8**
    """
    total_migrations = len(_MIGRATIONS)

    # Create a fresh in-memory SQLite database for each test case.
    db = DatabaseManager("sqlite:///:memory:")
    runner = MigrationRunner(db)

    # ----------------------------------------------------------------
    # Step 1: Bring the schema up to `starting_version` by running
    # migrations up to that point directly (simulates a DB that was
    # previously migrated to an intermediate version).
    # ----------------------------------------------------------------
    if starting_version > 0:
        # Apply only the migrations whose version <= starting_version.
        from sqlalchemy import text

        # First, run the subset of migrations to reach starting_version.
        # We do this by temporarily patching _MIGRATIONS isn't needed —
        # instead we manually apply each migration up to starting_version.
        from luma.storage.migrations import _MIGRATIONS as migrations_list

        for version, upgrade_fn in sorted(migrations_list, key=lambda t: t[0]):
            if version > starting_version:
                break
            with db.get_session() as session:
                upgrade_fn(session)
                # Upsert the schema_version row.
                session.execute(text("DELETE FROM schema_version"))
                session.execute(
                    text("INSERT INTO schema_version (version) VALUES (:v)"),
                    {"v": version},
                )
    else:
        # starting_version == 0: no migrations applied yet.
        # The schema_version table may not exist; that is fine — the runner
        # treats a missing table as version 0.
        pass

    # Confirm the current version matches the desired starting point.
    current_before = runner.get_current_version()
    assert current_before == starting_version, (
        f"Expected starting version {starting_version}, "
        f"got {current_before}"
    )

    # ----------------------------------------------------------------
    # Step 2: Run all pending migrations.
    # ----------------------------------------------------------------
    runner.run_pending()

    # ----------------------------------------------------------------
    # Step 3: Verify the final version equals the total migration count.
    # ----------------------------------------------------------------
    final_version = runner.get_current_version()
    assert final_version == total_migrations, (
        f"After run_pending() from version {starting_version}, "
        f"expected schema_version={total_migrations}, got {final_version}"
    )

    # ----------------------------------------------------------------
    # Step 4: Idempotency — calling run_pending() again must not change
    # the version (no migration is applied more than once).
    # ----------------------------------------------------------------
    runner.run_pending()
    version_after_second_run = runner.get_current_version()
    assert version_after_second_run == total_migrations, (
        f"After second run_pending(), expected schema_version={total_migrations}, "
        f"got {version_after_second_run}"
    )


import math

from luma.storage.repositories.memory_repository import MemoryRepository

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

valid_user_id = st.text(min_size=1, max_size=64).filter(str.strip)
valid_content = st.text(min_size=1, max_size=10_000)
valid_score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
valid_namespace = st.one_of(st.none(), st.text(min_size=1, max_size=64))


# ---------------------------------------------------------------------------
# Property 1: Memory round-trip
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 1: Memory round-trip
@given(
    user_id=valid_user_id,
    namespace=valid_namespace,
    content=valid_content,
    importance_score=valid_score,
    final_score=valid_score,
)
@settings(max_examples=100)
def test_memory_round_trip(user_id, namespace, content, importance_score, final_score):
    """
    Property: For any valid user_id, namespace, content, importance_score, final_score,
    create() then get_by_id() returns an equal MemoryRecord.

    **Validates: Requirements 5.2, 5.3, 16.4**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = MemoryRepository(session)
        record = repo.create(user_id, namespace, content, importance_score, final_score)
        retrieved = repo.get_by_id(record.id)

    assert retrieved is not None
    assert retrieved.user_id == user_id
    assert retrieved.namespace == namespace
    assert retrieved.content == content
    assert math.isclose(retrieved.importance_score, importance_score)
    assert math.isclose(retrieved.final_score, final_score)


# ---------------------------------------------------------------------------
# Property 2: Memory retrieval completeness and ordering
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 2: Memory retrieval completeness and ordering
@given(
    memories=st.lists(
        st.tuples(valid_user_id, valid_namespace, valid_content, valid_score, valid_score),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_memory_retrieval_completeness_and_ordering(memories):
    """
    Property: For any non-empty list of memories for a user_id, get_by_user()
    returns all records ordered by created_at descending.

    **Validates: Requirements 5.4**
    """
    # Use a fixed user_id so all memories belong to the same user.
    fixed_user_id = "test-ordering-user"

    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = MemoryRepository(session)
        created_ids = []
        for _, namespace, content, importance_score, final_score in memories:
            record = repo.create(fixed_user_id, namespace, content, importance_score, final_score)
            created_ids.append(record.id)

        results = repo.get_by_user(fixed_user_id, limit=len(memories) + 1)

    # All created records must be returned.
    assert len(results) == len(memories)
    assert set(r.id for r in results) == set(created_ids)

    # Records must be ordered by created_at descending (newest first).
    for i in range(len(results) - 1):
        assert results[i].created_at >= results[i + 1].created_at


# ---------------------------------------------------------------------------
# Property 3: Memory update round-trip
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 3: Memory update round-trip
@given(
    user_id=valid_user_id,
    namespace=valid_namespace,
    content=valid_content,
    original_importance=valid_score,
    original_final=valid_score,
    new_importance=valid_score,
    new_final=valid_score,
)
@settings(max_examples=100)
def test_memory_update_round_trip(
    user_id, namespace, content, original_importance, original_final, new_importance, new_final
):
    """
    Property: For any existing memory and valid (importance_score, final_score),
    update() then get_by_id() returns record with updated scores.

    **Validates: Requirements 5.5**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = MemoryRepository(session)
        record = repo.create(user_id, namespace, content, original_importance, original_final)
        updated = repo.update(record.id, importance_score=new_importance, final_score=new_final)
        retrieved = repo.get_by_id(record.id)

    assert updated is not None
    assert math.isclose(updated.importance_score, new_importance)
    assert math.isclose(updated.final_score, new_final)

    assert retrieved is not None
    assert math.isclose(retrieved.importance_score, new_importance)
    assert math.isclose(retrieved.final_score, new_final)


# ---------------------------------------------------------------------------
# Property 4: Memory delete consistency
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 4: Memory delete consistency
@given(
    user_id=valid_user_id,
    namespace=valid_namespace,
    content=valid_content,
    importance_score=valid_score,
    final_score=valid_score,
)
@settings(max_examples=100)
def test_memory_delete_consistency(user_id, namespace, content, importance_score, final_score):
    """
    Property: For any existing memory, delete() returns True and subsequent
    get_by_id() returns None. delete() on a non-existent id returns False.

    **Validates: Requirements 5.6, 16.8**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = MemoryRepository(session)
        record = repo.create(user_id, namespace, content, importance_score, final_score)
        memory_id = record.id

        # Delete existing record — must return True.
        result = repo.delete(memory_id)
        assert result is True

        # Subsequent get_by_id must return None.
        after_delete = repo.get_by_id(memory_id)
        assert after_delete is None

        # Deleting again (non-existent) must return False.
        result_again = repo.delete(memory_id)
        assert result_again is False


# ---------------------------------------------------------------------------
# Property 5: Memory count invariant
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 5: Memory count invariant
@given(
    user_id=valid_user_id,
    memories=st.lists(
        st.tuples(valid_namespace, valid_content, valid_score, valid_score),
        min_size=0,
        max_size=20,
    ),
)
@settings(max_examples=100)
def test_memory_count_invariant(user_id, memories):
    """
    Property: For any user_id and N create() calls, count_by_user() equals N.

    **Validates: Requirements 5.7**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = MemoryRepository(session)
        for namespace, content, importance_score, final_score in memories:
            repo.create(user_id, namespace, content, importance_score, final_score)
        count = repo.count_by_user(user_id)

    assert count == len(memories)


from luma.storage.repositories.insight_repository import InsightRepository

# ---------------------------------------------------------------------------
# Insight strategies
# ---------------------------------------------------------------------------

valid_evidence = st.one_of(st.none(), st.dictionaries(st.text(), st.text()))

# ---------------------------------------------------------------------------
# Property 6: Insight round-trip
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 6: Insight round-trip
@given(
    user_id=valid_user_id,
    message=valid_content,
    confidence=valid_score,
    evidence=valid_evidence,
)
@settings(max_examples=100)
def test_insight_round_trip(user_id, message, confidence, evidence):
    """
    Property: For any valid user_id, message, confidence, evidence,
    create() then get_by_id() returns an equal InsightRecord.

    **Validates: Requirements 6.2, 6.4, 16.5**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = InsightRepository(session)
        record = repo.create(user_id, message, confidence, evidence)
        retrieved = repo.get_by_id(record.id)

    assert retrieved is not None
    assert retrieved.user_id == user_id
    assert retrieved.message == message
    assert math.isclose(retrieved.confidence, confidence)
    assert retrieved.evidence == evidence


# ---------------------------------------------------------------------------
# Property 7: Insight retrieval ordering
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 7: Insight retrieval ordering
@given(
    insights=st.lists(
        st.tuples(valid_content, valid_score, valid_evidence),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=100)
def test_insight_retrieval_ordering(insights):
    """
    Property: For any non-empty list of insights for a user_id,
    get_by_user() returns all records ordered by created_at descending.

    **Validates: Requirements 6.3**
    """
    fixed_user_id = "test-insight-ordering-user"

    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = InsightRepository(session)
        created_ids = []
        for message, confidence, evidence in insights:
            record = repo.create(fixed_user_id, message, confidence, evidence)
            created_ids.append(record.id)

        results = repo.get_by_user(fixed_user_id, limit=len(insights) + 1)

    assert len(results) == len(insights)
    assert set(r.id for r in results) == set(created_ids)

    for i in range(len(results) - 1):
        assert results[i].created_at >= results[i + 1].created_at


from luma.storage.repositories.personalization_repository import PersonalizationRepository

# ---------------------------------------------------------------------------
# Personalization strategies
# ---------------------------------------------------------------------------

valid_interests = st.lists(st.text(min_size=1), max_size=20)
valid_preferences = st.dictionaries(st.text(min_size=1), st.text(), max_size=20)
valid_strengths = st.lists(st.text(min_size=1), max_size=20)


# ---------------------------------------------------------------------------
# Property 8: User profile round-trip
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 8: User profile round-trip
@given(
    user_id=valid_user_id,
    interests=valid_interests,
    preferences=valid_preferences,
    strengths=valid_strengths,
)
@settings(max_examples=100)
def test_user_profile_round_trip(user_id, interests, preferences, strengths):
    """
    Property: For any valid user_id, interests, preferences, strengths,
    upsert() then get_by_user() returns a UserProfileRecord reflecting the
    upserted values.

    **Validates: Requirements 7.2, 7.3, 16.6**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = PersonalizationRepository(session)
        repo.upsert(user_id, interests=interests, preferences=preferences, strengths=strengths)
        retrieved = repo.get_by_user(user_id)

    assert retrieved is not None
    assert retrieved.user_id == user_id
    assert retrieved.interests == interests
    assert retrieved.preferences == preferences
    assert retrieved.strengths == strengths


# ---------------------------------------------------------------------------
# Property 9: Partial upsert preserves existing fields
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 9: Partial upsert preserves existing fields
@given(
    user_id=valid_user_id,
    interests=valid_interests,
    preferences=valid_preferences,
    strengths=valid_strengths,
    omit_interests=st.booleans(),
    omit_preferences=st.booleans(),
    omit_strengths=st.booleans(),
)
@settings(max_examples=100)
def test_partial_upsert_preserves_existing_fields(
    user_id, interests, preferences, strengths,
    omit_interests, omit_preferences, omit_strengths,
):
    """
    Property: For any existing UserProfileRecord and any subset of fields
    passed as None to a subsequent upsert(), the None fields retain their
    pre-upsert values.

    **Validates: Requirements 7.5**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = PersonalizationRepository(session)
        # Initial upsert — establish baseline values.
        repo.upsert(user_id, interests=interests, preferences=preferences, strengths=strengths)

        # Partial upsert — pass None for omitted fields.
        repo.upsert(
            user_id,
            interests=None if omit_interests else interests,
            preferences=None if omit_preferences else preferences,
            strengths=None if omit_strengths else strengths,
        )

        retrieved = repo.get_by_user(user_id)

    assert retrieved is not None
    # Fields passed as None must retain their pre-upsert values.
    if omit_interests:
        assert retrieved.interests == interests
    if omit_preferences:
        assert retrieved.preferences == preferences
    if omit_strengths:
        assert retrieved.strengths == strengths


# ---------------------------------------------------------------------------
# Property 13: Upsert uniqueness under concurrent writes
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 13: Upsert uniqueness under concurrent writes
@given(
    user_id=valid_user_id,
    upserts=st.lists(
        st.tuples(valid_interests, valid_preferences, valid_strengths),
        min_size=1,
        max_size=10,
    ),
)
@settings(max_examples=100)
def test_upsert_uniqueness(user_id, upserts):
    """
    Property: For any user_id, after any number of upsert() calls for that
    user_id, the user_profiles table contains exactly one row for that user_id.

    **Validates: Requirements 14.2**
    """
    from sqlalchemy import text

    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = PersonalizationRepository(session)
        for interests, preferences, strengths in upserts:
            repo.upsert(user_id, interests=interests, preferences=preferences, strengths=strengths)

        # Count rows directly in the table for the given user_id.
        row_count = session.execute(
            text("SELECT COUNT(*) FROM user_profiles WHERE user_id = :uid"),
            {"uid": user_id},
        ).scalar()

    assert row_count == 1, (
        f"Expected exactly 1 row for user_id '{user_id}' after {len(upserts)} upserts, "
        f"got {row_count}"
    )


from luma.storage.repositories.teacher_repository import TeacherRepository

# ---------------------------------------------------------------------------
# Teacher strategies
# ---------------------------------------------------------------------------

valid_topic = st.text(min_size=1, max_size=128).filter(str.strip)
valid_progress = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
valid_weak_areas = st.lists(st.text(min_size=1), max_size=20)


# ---------------------------------------------------------------------------
# Property 10: Learning progress round-trip
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 10: Learning progress round-trip
@given(
    user_id=st.text(min_size=1, max_size=64).filter(str.strip),
    topic=valid_topic,
    progress=valid_progress,
    weak_areas=valid_weak_areas,
)
@settings(max_examples=100)
def test_learning_progress_round_trip(user_id, topic, progress, weak_areas):
    """
    Property: For any valid user_id, topic, progress, weak_areas,
    upsert_progress() then get_progress() returns a LearningProgressRecord
    reflecting the upserted values.

    **Validates: Requirements 8.2, 8.3, 16.7**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = TeacherRepository(session)
        repo.upsert_progress(user_id, topic, progress, weak_areas)
        retrieved = repo.get_progress(user_id, topic)

    assert retrieved is not None
    assert retrieved.user_id == user_id
    assert retrieved.topic == topic
    assert math.isclose(retrieved.progress, progress)
    assert retrieved.weak_areas == weak_areas


# ---------------------------------------------------------------------------
# Property 11: Learning progress completeness
# ---------------------------------------------------------------------------

# Feature: luma-persistence-storage, Property 11: Learning progress completeness
@given(
    user_id=st.text(min_size=1, max_size=64).filter(str.strip),
    topics=st.lists(
        st.text(min_size=1, max_size=64).filter(str.strip),
        min_size=1,
        max_size=10,
        unique=True,
    ),
)
@settings(max_examples=100)
def test_learning_progress_completeness(user_id, topics):
    """
    Property: For any set of N progress records for a user_id across distinct
    topics, get_all_progress() returns exactly N records.

    **Validates: Requirements 8.4**
    """
    db = DatabaseManager("sqlite:///:memory:")
    MigrationRunner(db).run_pending()
    with db.get_session() as session:
        repo = TeacherRepository(session)
        for topic in topics:
            repo.upsert_progress(user_id, topic, 0.5, [])
        results = repo.get_all_progress(user_id)

    assert len(results) == len(topics)
