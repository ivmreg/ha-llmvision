import aiosqlite
import shutil
import datetime
import uuid
import os, re
import json
import asyncio
from .const import DOMAIN, CONF_RETENTION_TIME, CONF_TIMELINE_LANGUAGE
from homeassistant.util import dt as dt_util
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from functools import partial
import logging

_LOGGER = logging.getLogger(__name__)

DB_VERSION = 5
DATA_TIMELINE_MUTATION_LOCKS = "llmvision_timeline_mutation_locks"
DATA_TIMELINE_STARTUP_TASKS = "llmvision_timeline_startup_tasks"


async def _get_category_and_label(
    hass: HomeAssistant, config_entry: ConfigEntry, query: str
) -> tuple[str, str]:
    """Return (category, label) for the best match in the query using the language regex template.
    - category: top-level category key (e.g., 'people', 'vehicles', ...)
    - label: canonical label mapped from the matched synonym (e.g., 'person', 'car', ...)
    Returns ("", "") when no match is found.
    """

    def load_lang_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Determine language (fallback to 'en')
    language = config_entry.options.get(CONF_TIMELINE_LANGUAGE) or "English"

    lang_map = {
        "Bulgarian": "bg",
        "Catalan": "ca",
        "Czech": "cs",
        "Danish": "da",
        "German": "de",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "Greek": "el",
        "Hungarian": "hu",
        "Italian": "it",
        "Dutch": "nl",
        "Polish": "pl",
        "Portuguese": "pt",
        "Slovak": "sk",
        "Swedish": "sv",
    }
    lang_code = lang_map.get(language, "en")

    # Load language json from /timeline_strings/{language}.json
    lang_file_path = os.path.join(
        os.path.dirname(__file__), "timeline_strings", f"{lang_code}.json"
    )
    if not os.path.exists(lang_file_path):
        _LOGGER.warning(
            f"Language file {lang_file_path} does not exist. Defaulting to English."
        )
        lang_file_path = os.path.join(
            os.path.dirname(__file__), "timeline_strings", "en.json"
        )

    try:
        lang_data = await hass.async_add_executor_job(load_lang_json, lang_file_path)
    except Exception as e:
        _LOGGER.error(f"Error loading language file {lang_file_path}: {e}")
        return ("", "")

    categories_data = lang_data.get("categories", {})
    regex_template = lang_data.get("regex")

    def compile_pattern(key: str):
        """Compile a regex pattern for a given key using the optional template in the JSON.
        Supports:
          - string like: "`\\b${key}s?\\b`, 'i'"
          - object like: {"pattern": "\\b${key}s?\\b", "flags": "i"}
        Falls back to: r"\b{key}s?\b" with IGNORECASE.
        """
        # Defaults: ensure whole-word matching (word boundaries)
        pattern = rf"\b{re.escape(key)}s?\b"
        flags = re.IGNORECASE

        def flags_from_str(f: str) -> int:
            f = (f or "").lower()
            fl = 0
            if "i" in f:
                fl |= re.IGNORECASE
            if "m" in f:
                fl |= re.MULTILINE
            if "s" in f:
                fl |= re.DOTALL
            return fl

        try:
            if isinstance(regex_template, str):
                # Expect format: "`\\b${key}s?\\b`, 'i'" or similar
                m = re.search(r"`([^`]*)`(?:\s*,\s*'([a-zA-Z]+)')?", regex_template)
                if m:
                    tpl = m.group(1)
                    tpl_f = m.group(2) or ""
                    # Substitute ${key} placeholder safely (escape key)
                    pattern = tpl.replace("${key}", re.escape(key))
                    flags = flags_from_str(tpl_f)
                    # Ensure whole-word boundaries around the key if not present
                    if not re.search(r"\\b", pattern):
                        pattern = rf"\\b{re.escape(key)}s?\\b"
            elif isinstance(regex_template, dict):
                tpl_pat = regex_template.get("pattern")
                tpl_flags = regex_template.get("flags", "")
                if isinstance(tpl_pat, str):
                    pattern = tpl_pat.replace("${key}", re.escape(key))
                    flags = flags_from_str(tpl_flags)
                    if not re.search(r"\\b", pattern):
                        pattern = rf"\\b{re.escape(key)}s?\\b"
        except Exception as e:
            _LOGGER.debug(f"Failed to apply regex template for key '{key}': {e}")

        try:
            return re.compile(pattern, flags)
        except re.error as e:
            _LOGGER.warning(f"Invalid regex for key '{key}': {pattern} ({e})")
            return None

    # Collect all matches so we can pick the best according to category priority
    # Priority order (lower is higher priority)
    category_priority = {
        "delivery": 0,
        "vehicle": 1,
        "person": 2,
        "animal": 3,
        "entity": 4,
        "nature": 5,
    }

    matches: list[tuple[int, int, int, str, str]] = (
        []
    )  # (priority, -key_len, order_index, category, label)
    q = query or ""

    for cat_name, cat_def in categories_data.items():
        objects = (cat_def or {}).get("labels", {})
        if not isinstance(objects, dict):
            continue

        for key_index, (key, canonical) in enumerate(objects.items()):
            pat = compile_pattern(str(key))
            if pat and pat.search(q):
                canonical_label = str(canonical) if canonical else str(key)
                prio = category_priority.get(str(cat_name).lower(), 99)
                matches.append(
                    (prio, -len(str(key)), key_index, str(cat_name), canonical_label)
                )

    if not matches:
        return ("", "")

    # Sort by category priority (asc), then longest key (desc via negative length), then original order
    matches.sort(key=lambda x: (x[0], x[1], x[2]))
    _, _, _, category, label = matches[0]
    return (category, label)


class Event:

    def __init__(
        self,
        uid: str,
        title: str,
        start: datetime.datetime | None,
        end: datetime.datetime | None,
        description: str,
        key_frame: str,
        camera_name: str,
        category: str,
        label: str,
        saved_frames: list[str] | None = None,
    ):
        self.uid = uid
        self.title = title
        self.start = start
        self.end = end
        self.description = description
        self.key_frame = key_frame
        self.camera_name = camera_name
        self.category = category
        self.label = label
        if saved_frames is not None:
            frames = list(saved_frames)
            if key_frame and key_frame not in frames:
                frames.append(key_frame)
            self.saved_frames = frames
        elif key_frame:
            self.saved_frames = [key_frame]
        else:
            self.saved_frames = []

    def __repr__(self):
        return f"Event(uid={self.uid}, title={self.title}, start={self.start}, end={self.end}, description={self.description}, key_frame={self.key_frame}, camera_name={self.camera_name}, category={self.category}, label={self.label}, saved_frames={self.saved_frames})"

    def __str__(self):
        return self.__repr__()


class Timeline:
    """Representation of a Calendar."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        """Initialize the calendar"""
        self.hass = hass
        self.events: list[Event] = []
        self._events_loaded = False
        self.today_summary = ""
        self.retention_time = config_entry.data.get(CONF_RETENTION_TIME)

        # Track key_frame paths whose DB rows are not yet committed
        self._pending_key_frames: set[str] = set()
        self._fallback_lock = asyncio.Lock()
        self._config_entry = config_entry
        self._migrating = True

        # Path to the JSON file where events are stored
        self._db_path = os.path.join(self.hass.config.path("llmvision"), "events.db")
        self._media_path = f"/media/llmvision/snapshots"
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        os.makedirs(self._media_path, exist_ok=True)

        self._startup_task = None
        startup_tasks = self.hass.data.setdefault(DATA_TIMELINE_STARTUP_TASKS, {})
        canonical_db = os.path.normcase(os.path.abspath(self._db_path))
        task = startup_tasks.get(canonical_db)
        if task is None or (
            task.done() and (task.cancelled() or task.exception() is not None)
        ):
            task = self.hass.async_create_task(self._startup())
            startup_tasks[canonical_db] = task
        self._startup_task = task

    async def _startup(self) -> None:
        """Create and migrate this database before any ordinary operation."""
        await self._initialize_db()
        if not await self._migrate():
            raise RuntimeError("LLM Vision timeline database migration failed")
        await self.load_events()

    async def _ensure_ready(self) -> None:
        """Wait for shared startup, without making migration await itself."""
        task = self._startup_task
        if task is not None and task is not asyncio.current_task():
            # A cancelled service request must not cancel startup for other users.
            await asyncio.shield(task)
            self._migrating = False

    def _get_mutation_lock(self) -> asyncio.Lock:
        """Return the shared mutation lock for this hass instance and db_path."""
        if not hasattr(self.hass, "data") or not isinstance(self.hass.data, dict):
            return self._fallback_lock
        locks = self.hass.data.setdefault(DATA_TIMELINE_MUTATION_LOCKS, {})
        canonical_db = os.path.normcase(os.path.abspath(self._db_path))
        if canonical_db not in locks:
            locks[canonical_db] = asyncio.Lock()
        return locks[canonical_db]

    @property
    def _cleanup_lock(self) -> asyncio.Lock:
        return self._get_mutation_lock()

    @_cleanup_lock.setter
    def _cleanup_lock(self, val: asyncio.Lock) -> None:
        self._fallback_lock = val

    def _normalize_image_path(self, path: str) -> str:
        """Normalize internal image path for canonical comparison and ownership."""
        if not path or not isinstance(path, str):
            return ""
        try:
            return os.path.normcase(os.path.realpath(os.path.abspath(path)))
        except (ValueError, OSError):
            return os.path.normcase(os.path.normpath(path))

    async def _get_db_version(self) -> int:
        """Return PRAGMA user_version (0 if unset)."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute("PRAGMA user_version") as cur:
                    row = await cur.fetchone()
                    if not row or row[0] is None:
                        return 0
                    return int(row[0])
        except Exception as e:
            _LOGGER.debug(f"Failed to read DB user_version: {e}")
            return 0

    async def _set_db_version(self, version: int) -> None:
        """Set PRAGMA user_version."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(f"PRAGMA user_version = {int(version)}")
                await db.commit()
        except Exception as e:
            _LOGGER.warning(f"Failed to set DB user_version to {version}: {e}")

    async def _has_latest_schema(self) -> bool:
        """Checks if the events and event_images tables match the latest schema."""
        expected = {
            "uid",
            "title",
            "start",
            "end",
            "description",
            "key_frame",
            "camera_name",
            "category",
            "label",
        }
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute("PRAGMA table_info(events)") as cursor:
                    cols = [r[1] for r in await cursor.fetchall()]
                if not expected.issubset(set(cols)):
                    return False
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='event_images'"
                ) as cursor:
                    if not await cursor.fetchone():
                        return False
            return True
        except Exception as e:
            _LOGGER.debug(f"Failed to inspect DB schema: {e}")
            return False

    async def _initialize_db(self):
        """Initialize database"""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        uid TEXT PRIMARY KEY,
                        title TEXT,
                        start TEXT,
                        end TEXT,
                        description TEXT,
                        key_frame TEXT,
                        camera_name TEXT,
                        category TEXT,
                        label TEXT
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_start ON events (start)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_end ON events (end)
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS event_images (
                        event_uid TEXT,
                        image_path TEXT,
                        PRIMARY KEY (event_uid, image_path)
                    )
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_images_uid ON event_images (event_uid)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_event_images_path ON event_images (image_path)
                """)
                await db.commit()
        except aiosqlite.Error as e:
            _LOGGER.error(f"Error initializing database: {e}")
            raise
        # Creating tables does not backfill legacy images or import events.json.
        # Only successful migration may advance user_version.

    async def _migrate(self):
        """Migrate events.db, leaving failed migrations retryable."""
        completed = False
        try:
            current_version = await self._get_db_version()
            old_db_path = os.path.join(
                self.hass.config.path("llmvision"), "events.json"
            )
            if current_version >= DB_VERSION and (
                current_version > DB_VERSION or not os.path.exists(old_db_path)
            ):
                if current_version == DB_VERSION:
                    # Earlier initialization could stamp v5 before backfill ran.
                    await self._backfill_event_images()
                completed = await self._has_latest_schema()
                return completed

            _LOGGER.info(
                f"Starting DB migration (user_version={current_version} -> {DB_VERSION})"
            )
            # v1 -> v2: Migrate events from events.json to events.db
            if os.path.exists(old_db_path):
                _LOGGER.info("Migrating events from events.json to events.db")
                with open(old_db_path, "r") as file:
                    data = json.load(file)
                    event_counter = 0
                    for event in data:
                        # Stay in the startup task so readiness does not wait on
                        # the very migration importing these legacy events.
                        await self.create_event(
                            start=datetime.datetime.fromisoformat(event["start"]),
                            end=datetime.datetime.fromisoformat(event["end"]),
                            title=event.get("summary", ""),
                            description=event.get("description", ""),
                            key_frame=event["location"].split(",")[0],
                            camera_name=(
                                event["location"].split(",")[1]
                                if len(event["location"].split(",")) > 1
                                else ""
                            ),
                            label=event.get("label", ""),
                        )
                        event_counter += 1
                    _LOGGER.info(f"Migrated {event_counter} events")
                _LOGGER.info("Migration complete, deleting events.json")
                os.remove(old_db_path)

            # v3 -> v4: Migrate image paths to /config/media/llmvision/snapshots from /www/llmvision
            try:
                # Move images to new location
                # Ensure dir exists
                await self.hass.loop.run_in_executor(
                    None,
                    partial(
                        os.makedirs,
                        self.hass.config.path("media/llmvision/snapshots"),
                        exist_ok=True,
                    ),
                )
                src_dir = self.hass.config.path("www/llmvision")
                dst_dir = self.hass.config.path("media/llmvision/snapshots")
                if os.path.exists(src_dir):
                    for filename in await self.hass.loop.run_in_executor(
                        None, partial(os.listdir, src_dir)
                    ):
                        src_file = os.path.join(src_dir, filename)
                        dst_file = os.path.join(dst_dir, filename)
                        if os.path.isfile(src_file):
                            await self.hass.loop.run_in_executor(
                                None, shutil.move, src_file, dst_file
                            )

                async with aiosqlite.connect(self._db_path) as db:
                    await db.execute("""
                        UPDATE events SET key_frame = REPLACE(key_frame, '/www/llmvision', '/media/llmvision/snapshots')
                    """)
                    await db.commit()
            except aiosqlite.Error as e:
                _LOGGER.error(f"Error migrating events.db to v4: {e}")
                return False

            # v4 -> v4.1: Migrate image paths to /media/llmvision/snapshots from /config/media/llmvision/snapshots
            try:
                # Move images to new location
                # Ensure dir exists
                await self.hass.loop.run_in_executor(
                    None,
                    partial(
                        os.makedirs,
                        "/media/llmvision/snapshots",
                        exist_ok=True,
                    ),
                )
                src_dir = self.hass.config.path("media/llmvision/snapshots")
                dst_dir = "/media/llmvision/snapshots"
                if os.path.exists(src_dir):
                    for filename in await self.hass.loop.run_in_executor(
                        None, partial(os.listdir, src_dir)
                    ):
                        src_file = os.path.join(src_dir, filename)
                        dst_file = os.path.join(dst_dir, filename)
                        if os.path.isfile(src_file):
                            await self.hass.loop.run_in_executor(
                                None, shutil.move, src_file, dst_file
                            )

                async with aiosqlite.connect(self._db_path) as db:
                    await db.execute("""
                        UPDATE events SET key_frame = REPLACE(key_frame, '/config/media/llmvision/snapshots', '/media/local/llmvision/snapshots')
                    """)
                    await db.commit()
            except aiosqlite.Error as e:
                _LOGGER.error(f"Error migrating events.db to v4.1: {e}")
                return False

            # v4.1 -> v4.2: Add category column to events.db
            try:
                async with aiosqlite.connect(self._db_path) as db:
                    async with db.execute("""PRAGMA table_info(events)""") as cursor:
                        columns = await cursor.fetchall()
                        column_names = [column[1] for column in columns]
                        if "category" not in column_names:
                            _LOGGER.info(
                                "Migrating events.db to include category column"
                            )
                            await db.execute(
                                """ALTER TABLE events ADD COLUMN category TEXT"""
                            )
                            await db.commit()
                            _LOGGER.info("Timeline DB migration to v4.2 complete")
            except aiosqlite.Error as e:
                _LOGGER.error(f"Error migrating events.db to v4.2: {e}")
                return False

            # v4.2 -> v4.3: Remove today_summary column from events.db, add label column, populate label column, rename summary->title
            try:
                async with aiosqlite.connect(self._db_path) as db:
                    async with db.execute("""PRAGMA table_info(events)""") as cursor:
                        columns = await cursor.fetchall()
                        column_names = [column[1] for column in columns]
                        if "today_summary" in column_names:
                            _LOGGER.info("Removing today_summary column from events.db")
                            await db.execute(
                                """ALTER TABLE events DROP COLUMN today_summary"""
                            )
                            await db.commit()
                        if "label" not in column_names:
                            _LOGGER.info("Adding label column to events.db")
                            await db.execute(
                                """ALTER TABLE events ADD COLUMN label TEXT"""
                            )
                            await db.commit()
                        if "summary" in column_names:
                            _LOGGER.info("Renaming summary column to title")
                            await db.execute(
                                """ALTER TABLE events RENAME TO events_old"""
                            )
                            await db.execute("""
                                CREATE TABLE events (
                                    uid TEXT PRIMARY KEY,
                                    title TEXT,
                                    start TEXT,
                                    end TEXT,
                                    description TEXT,
                                    key_frame TEXT,
                                    camera_name TEXT,
                                    category TEXT,
                                    label TEXT
                                )
                            """)
                            await db.execute("""
                                INSERT INTO events (uid, title, start, end, description, key_frame, camera_name, category, label)
                                SELECT uid, summary, start, end, description, key_frame, camera_name, category, label
                                FROM events_old
                            """)
                            await db.execute("""DROP TABLE events_old""")
                            await db.commit()
                            _LOGGER.info("Renamed summary column to title")
                        if "label" in column_names:
                            _LOGGER.info(
                                "Populating label and category columns in events.db"
                            )
                            async with db.execute(
                                """SELECT uid, title FROM events WHERE label IS NULL OR label = '' OR category IS NULL OR category = ''"""
                            ) as cursor:
                                rows = await cursor.fetchall()
                                for row in rows:
                                    uid = row[0]
                                    title = row[1]
                                    category, label = await _get_category_and_label(
                                        self.hass, self._config_entry, title
                                    )
                                    await db.execute(
                                        """UPDATE events SET label = ? WHERE uid = ?""",
                                        (label, uid),
                                    )
                                    await db.execute(
                                        """UPDATE events SET category = ? WHERE uid = ?""",
                                        (category, uid),
                                    )
                            await db.commit()
            except aiosqlite.Error as e:
                _LOGGER.error(f"Error migrating events.db to v4.3: {e}")
                return False

            # v4.3 -> v5: Add event_images table and migrate existing key_frames
            try:
                await self._backfill_event_images()
            except aiosqlite.Error as e:
                _LOGGER.error(f"Error migrating events.db to v5: {e}")
                return False

            # Mark migration complete by setting user_version
            if not await self._has_latest_schema():
                return False
            await self._set_db_version(DB_VERSION)
            completed = await self._get_db_version() == DB_VERSION
            if not completed:
                return False
            _LOGGER.info(f"DB migration complete (user_version={DB_VERSION})")
            return True
        finally:
            self._migrating = not completed
            if completed:
                try:
                    await self._cleanup()
                except Exception as e:
                    _LOGGER.warning(f"Post-migration cleanup failed: {e}")

    async def _backfill_event_images(self) -> None:
        """Idempotently repair legacy keyframe associations, including stamped v5 DBs."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS event_images (
                    event_uid TEXT,
                    image_path TEXT,
                    PRIMARY KEY (event_uid, image_path)
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_images_uid ON event_images (event_uid)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_images_path ON event_images (image_path)
            """)
            await db.execute("""
                INSERT OR IGNORE INTO event_images (event_uid, image_path)
                SELECT uid, key_frame FROM events
                WHERE key_frame IS NOT NULL AND key_frame != ''
            """)
            await db.commit()
            _LOGGER.info("Timeline DB migration to v5 complete (event_images created)")

    def _ensure_datetime(self, dt):
        """Ensures the input is a datetime.datetime object"""
        if isinstance(dt, datetime.date) and not isinstance(dt, datetime.datetime):
            dt = datetime.datetime.combine(dt, datetime.datetime.min.time())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    def _get_retention_days(self) -> int | None:
        """Return the configured retention window in days (None disables cleanup)."""
        raw_value = self._config_entry.options.get(CONF_RETENTION_TIME)
        if raw_value is None:
            raw_value = self.retention_time

        if raw_value in (None, ""):
            return None

        try:
            days = int(raw_value)
        except (TypeError, ValueError):
            _LOGGER.warning(
                "Invalid retention_time=%s. Skipping automatic event purge.",
                raw_value,
            )
            return None

        if days <= 0:
            return None

        return days

    def _is_safe_snapshot_path(self, path: str) -> bool:
        """Returns True if path is safely inside the configured snapshots media directory."""
        if not path or not isinstance(path, str):
            return False
        try:
            abs_media = os.path.realpath(os.path.abspath(self._media_path))
            abs_path = os.path.realpath(os.path.abspath(path))
            common = os.path.commonpath([abs_media, abs_path])
            return (
                os.path.normcase(common) == os.path.normcase(abs_media)
                and os.path.normcase(abs_path) != os.path.normcase(abs_media)
            )
        except (ValueError, OSError):
            return False

    async def _purge_expired_events(self) -> None:
        """Remove events older than now - retention_time days and unshared media assets."""
        await self._ensure_ready()
        if getattr(self, "_migrating", False):
            return

        retention_days = self._get_retention_days()
        if retention_days is None:
            return

        cutoff = dt_util.utcnow() - datetime.timedelta(days=retention_days)
        cutoff_local = dt_util.as_local(self._ensure_datetime(cutoff)).isoformat()

        try:
            unshared_images_to_delete: list[str] = []
            async with self._get_mutation_lock():
                async with aiosqlite.connect(self._db_path) as db:
                    async with db.execute(
                        """
                        SELECT uid, key_frame FROM events
                        WHERE start IS NOT NULL AND start < ?
                        """,
                        (cutoff_local,),
                    ) as cursor:
                        stale_rows = list(await cursor.fetchall())

                    if not stale_rows:
                        return

                    # Check if event_images table exists
                    has_event_images = False
                    async with db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='event_images'"
                    ) as cur:
                        has_event_images = await cur.fetchone() is not None

                    expired_images = set()
                    for row in stale_rows:
                        if row[1]:
                            expired_images.add(row[1])

                    if has_event_images:
                        async with db.execute(
                            """
                            SELECT image_path FROM event_images
                            WHERE event_uid IN (SELECT uid FROM events WHERE start IS NOT NULL AND start < ?)
                            """,
                            (cutoff_local,),
                        ) as cur:
                            for img_row in await cur.fetchall():
                                if img_row[0]:
                                    expired_images.add(img_row[0])

                    # Fetch surviving events key_frames and event_images
                    async with db.execute(
                        """
                        SELECT key_frame FROM events
                        WHERE (start IS NULL OR start >= ?) AND key_frame IS NOT NULL AND key_frame != ''
                        """,
                        (cutoff_local,),
                    ) as cur:
                        surviving_keyframes = {
                            self._normalize_image_path(r[0])
                            for r in await cur.fetchall()
                        }

                    surviving_saved_frames = set()
                    if has_event_images:
                        async with db.execute(
                            """
                            SELECT ei.image_path FROM event_images ei
                            JOIN events e ON ei.event_uid = e.uid
                            WHERE (e.start IS NULL OR e.start >= ?) AND ei.image_path IS NOT NULL AND ei.image_path != ''
                            """,
                            (cutoff_local,),
                        ) as cur:
                            surviving_saved_frames = {
                                self._normalize_image_path(r[0])
                                for r in await cur.fetchall()
                            }

                    surviving_all = surviving_keyframes | surviving_saved_frames

                    for img in expired_images:
                        canon_img = self._normalize_image_path(img)
                        if canon_img and canon_img not in surviving_all:
                            unshared_images_to_delete.append(img)

                    # Delete associations and events transactionally
                    if has_event_images:
                        await db.execute(
                            """
                            DELETE FROM event_images
                            WHERE event_uid IN (SELECT uid FROM events WHERE start IS NOT NULL AND start < ?)
                            """,
                            (cutoff_local,),
                        )
                    await db.execute(
                        "DELETE FROM events WHERE start IS NOT NULL AND start < ?",
                        (cutoff_local,),
                    )
                    await db.commit()

                # Remove unshared filesystem assets after DB commit succeeds (still under lock)
                deleted_canonical = set()
                for img_path in unshared_images_to_delete:
                    canon = self._normalize_image_path(img_path)
                    if canon in deleted_canonical:
                        continue
                    if self._is_safe_snapshot_path(img_path):
                        try:
                            exists = await self.hass.async_add_executor_job(
                                os.path.exists, img_path
                            )
                            if exists:
                                await self.hass.async_add_executor_job(
                                    os.remove, img_path
                                )
                                deleted_canonical.add(canon)
                                _LOGGER.info(f"Purged expired unshared image: {img_path}")
                        except OSError as e:
                            _LOGGER.warning(f"Failed to purge expired image {img_path}: {e}")

            _LOGGER.info(
                "Purged %s expired timeline event(s) older than %s day(s)",
                len(stale_rows),
                retention_days,
            )
        except aiosqlite.Error as e:
            _LOGGER.error(f"Error purging expired timeline events: {e}")

    async def _get_category_from_label(self, label: str) -> str:
        """Returns the category for a given label using the language regex template."""
        return (await _get_category_and_label(self.hass, self._config_entry, label))[0]

    async def get_linked_images(self):
        """Returns the filenames of all images associated with events"""
        await self.load_events()
        linked = set()
        for e in self.events:
            if getattr(e, "key_frame", None):
                base = os.path.basename(e.key_frame)
                if base:
                    linked.add(base)
            for frame in getattr(e, "saved_frames", []):
                if frame:
                    base = os.path.basename(frame)
                    if base:
                        linked.add(base)
        return list(linked)

    async def load_events(self):
        """Loads events from the database into memory"""
        await self._ensure_ready()
        await self._purge_expired_events()
        self.events = []
        try:
            async with aiosqlite.connect(self._db_path) as db:
                has_event_images = False
                async with db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='event_images'"
                ) as cursor:
                    has_event_images = await cursor.fetchone() is not None

                event_frames_map: dict[str, list[str]] = {}
                if has_event_images:
                    async with db.execute(
                        "SELECT event_uid, image_path FROM event_images"
                    ) as cursor:
                        for ev_uid, img_path in await cursor.fetchall():
                            if ev_uid and img_path:
                                event_frames_map.setdefault(ev_uid, []).append(img_path)

                async with db.execute("""
                    SELECT
                        uid, title, start, end, description,
                        category, key_frame, camera_name, label
                    FROM events
                    """) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        # row: uid, summary, start, end, description, category, key_frame, camera_name, label
                        try:
                            row_start = (
                                dt_util.parse_datetime(row[2]) if row[2] else None
                            )
                            row_end = dt_util.parse_datetime(row[3]) if row[3] else None
                        except Exception:
                            # skip malformed rows
                            continue

                        if row_start:
                            row_start = self._ensure_datetime(row_start)
                        if row_end:
                            row_end = self._ensure_datetime(row_end)

                        uid = row[0]
                        saved_frames = event_frames_map.get(uid)

                        event = Event(
                            uid=uid,
                            title=row[1],
                            start=row_start,
                            end=row_end,
                            description=row[4],
                            category=row[5],
                            key_frame=row[6],
                            camera_name=row[7],
                            label=row[8],
                            saved_frames=saved_frames,
                        )
                        self.events.append(event)
            self._events_loaded = True
        except aiosqlite.Error as e:
            _LOGGER.error(f"Error loading events from database: {e}")

    async def get_all_events(self) -> list[Event]:
        """Returns calendar events"""
        await self._ensure_ready()
        if not self._events_loaded:
            await self.load_events()
        return self.events

    async def get_event(self, uid: str) -> dict | None:
        """Returns a single event by UID as a dict. Used by the API."""
        await self.load_events()
        for event in self.events:
            if event.uid == uid:
                # return event as dict
                event_dict = {
                    "uid": event.uid,
                    "title": event.title,
                    "start": event.start.isoformat() if event.start else None,
                    "end": event.end.isoformat() if event.end else None,
                    "description": event.description,
                    "key_frame": event.key_frame,
                    "camera_name": event.camera_name,
                    "category": event.category,
                    "label": event.label,
                    "saved_frames": getattr(event, "saved_frames", []),
                }
                return event_dict
        return None

    async def get_events_json(
        self,
        limit=100,
        cameras=[],
        categories=[],
        labels=[],
        start=None,
        end=None,
        include_no_activity=False,
    ) -> list[dict]:
        """Returns json event data from the database. Used by the API.
        Supports filtering by camera, category, time range, and sorting by start (newest first).
        Set include_no_activity=True to include events whose title is 'no activity observed'.
        """
        _LOGGER.debug(
            f"Fetching events with filters - cameras: {cameras}, categories: {categories}, labels: {labels}, start: {start}, end: {end}, include_no_activity: {include_no_activity}"
        )
        await self._purge_expired_events()
        events: list[dict] = []

        # Normalize start/end inputs to timezone-aware datetimes (or None)
        def normalize_input_dt(dt_in):
            if dt_in is None:
                return None
            try:
                if isinstance(dt_in, str):
                    dt = dt_util.parse_datetime(dt_in)
                else:
                    dt = dt_in
                return self._ensure_datetime(dt)
            except Exception:
                return None

        start_dt = normalize_input_dt(start)
        end_dt = normalize_input_dt(end)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute("""
                SELECT
                    uid, title, start, end, description,
                    category, key_frame, camera_name, label
                FROM events
                """) as cursor:
                rows = await cursor.fetchall()
                for row in rows:
                    # row: uid, title, start, end, description, key_frame, camera_name, label
                    try:
                        row_start = dt_util.parse_datetime(row[2]) if row[2] else None
                        row_end = dt_util.parse_datetime(row[3]) if row[3] else None
                    except Exception:
                        # skip malformed rows
                        continue

                    if row_start:
                        row_start = self._ensure_datetime(row_start)
                    if row_end:
                        row_end = self._ensure_datetime(row_end)

                    # No-activity filter (always applied unless opted in)
                    if not include_no_activity:
                        if "no activity" in (row[1] or "").strip().lower():
                            continue

                    # Camera filter — events with no camera pass through unconditionally
                    if cameras and (row[7] or "").strip():
                        camera_name = (row[7] or "").lower()
                        if camera_name not in [c.lower() for c in cameras]:
                            continue

                    # Category filter
                    if categories:
                        category_name = (row[5] or "").lower()
                        if category_name not in [c.lower() for c in categories]:
                            continue

                    # Label filter
                    if labels:
                        label_name = (row[8] or "").lower()
                        if label_name not in [l.lower() for l in labels]:
                            continue

                    # Range overlap filter
                    if start_dt and row_end and row_end <= start_dt:
                        continue
                    if end_dt and row_start and row_start >= end_dt:
                        continue

                    events.append(
                        {
                            "uid": row[0],
                            "title": row[1],
                            "start": row[2],
                            "end": row[3],
                            "description": row[4],
                            "key_frame": row[6],
                            "camera_name": row[7],
                            "category": row[5],
                            "label": row[8],
                        }
                    )

        # Sort by start datetime descending (newest first). Malformed/missing start go to the end.
        def sort_key(ev):
            try:
                d = dt_util.parse_datetime(ev.get("start"))
                return dt_util.as_local(self._ensure_datetime(d))
            except Exception:
                return datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc)

        events.sort(key=sort_key, reverse=True)

        if limit is not None:
            return events[:limit]

        return events

    async def create_event(
        self,
        start: datetime.datetime,
        end: datetime.datetime,
        title: str,
        description: str,
        key_frame: str,
        camera_name: str,
        label: str = "",
        saved_frames: list[str] | None = None,
    ) -> None:
        """Adds a new event to calendar"""
        await self._ensure_ready()
        category = ""

        # Ensure dtstart and dtend are datetime objects
        if isinstance(start, str):
            start = datetime.datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.datetime.fromisoformat(end)

        start = dt_util.as_local(start)
        end = dt_util.as_local(end)

        # Mark key_frame and saved_frames as pending so cleanup won't remove them before DB insert
        pending_names = set()
        if key_frame:
            base = os.path.basename(key_frame)
            if base:
                pending_names.add(base.lower())
        if saved_frames:
            for frame in saved_frames:
                if frame:
                    base = os.path.basename(frame)
                    if base:
                        pending_names.add(base.lower())

        async with self._get_mutation_lock():
            for name in pending_names:
                self._pending_key_frames.add(name)
            try:
                # Resolve category and label if not provided
                if not label:
                    try:
                        query_text = " ".join(
                            part for part in (title or "", description or "") if part
                        )
                        auto_category, auto_label = await _get_category_and_label(
                            self.hass, self._config_entry, query_text
                        )
                        if not category:
                            category = auto_category
                        if not label:
                            label = auto_label
                    except Exception as e:
                        _LOGGER.warning(f"Failed to resolve category: {e}")
                else:
                    category = await self._get_category_from_label(label)

                event = Event(
                    uid=str(uuid.uuid4()),
                    title=title,
                    start=start,
                    end=end,
                    description=description,
                    key_frame=key_frame,
                    camera_name=camera_name,
                    category=category,
                    label=label,
                    saved_frames=saved_frames,
                )
                _LOGGER.info(f"Creating event: {event}")
                await self._insert_event(event)
            finally:
                # Remove from pending once DB insert is done
                for name in pending_names:
                    self._pending_key_frames.discard(name)
        await self.load_events()

    async def _insert_event(self, event: Event) -> None:
        """Inserts a new event into the database transactionally."""
        try:
            async with aiosqlite.connect(self._db_path) as db:
                _LOGGER.info(f"Inserting event into database: {event}")
                await db.execute(
                    """
                    INSERT INTO events (uid, title, start, end, description, key_frame, camera_name, category, label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.uid,
                        event.title,
                        dt_util.as_local(
                            self._ensure_datetime(event.start)
                        ).isoformat(),
                        dt_util.as_local(self._ensure_datetime(event.end)).isoformat(),
                        event.description,
                        event.key_frame,
                        event.camera_name,
                        event.category,
                        event.label,
                    ),
                )
                # Ensure schema ready: create event_images table transactionally
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS event_images (
                        event_uid TEXT,
                        image_path TEXT,
                        PRIMARY KEY (event_uid, image_path)
                    )
                """
                )
                images_to_insert = set()
                if event.key_frame:
                    norm_kf = self._normalize_image_path(event.key_frame)
                    if norm_kf:
                        images_to_insert.add(norm_kf)
                for frame in (getattr(event, "saved_frames", None) or []):
                    if frame:
                        norm_f = self._normalize_image_path(frame)
                        if norm_f:
                            images_to_insert.add(norm_f)
                if images_to_insert:
                    await db.executemany(
                        """
                        INSERT OR IGNORE INTO event_images (event_uid, image_path)
                        VALUES (?, ?)
                        """,
                        [(event.uid, img) for img in images_to_insert],
                    )
                await db.commit()
        except Exception as e:
            _LOGGER.error(f"Error inserting event into database: {e}")
            raise

    async def update_event(
        self,
        uid: str,
        start: datetime.datetime,
        end: datetime.datetime,
        title: str,
        description: str,
        key_frame: str,
        camera_name: str,
        label: str,
        saved_frames: list[str] | None = None,
    ) -> None:
        """Updates an existing event in the calendar."""
        await self._ensure_ready()
        # Ensure dtstart and dtend are datetime objects
        if isinstance(start, str):
            start = datetime.datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.datetime.fromisoformat(end)

        start = dt_util.as_local(start)
        end = dt_util.as_local(end)

        try:
            async with self._get_mutation_lock():
                async with aiosqlite.connect(self._db_path) as db:
                    cursor = await db.execute(
                        """
                        UPDATE events
                        SET title = ?, start = ?, end = ?, description = ?, key_frame = ?, camera_name = ?, category = ?, label = ?
                        WHERE uid = ?
                    """,
                        (
                            title,
                            dt_util.as_local(self._ensure_datetime(start)).isoformat(),
                            dt_util.as_local(self._ensure_datetime(end)).isoformat(),
                            description,
                            key_frame,
                            camera_name,
                            await self._get_category_from_label(label),
                            label,
                            uid,
                        ),
                    )
                    if cursor.rowcount > 0:
                        await db.execute(
                            """
                            CREATE TABLE IF NOT EXISTS event_images (
                                event_uid TEXT,
                                image_path TEXT,
                                PRIMARY KEY (event_uid, image_path)
                            )
                        """
                        )
                        if saved_frames is not None:
                            # Explicit replacement list given: replace associations
                            await db.execute(
                                "DELETE FROM event_images WHERE event_uid = ?", (uid,)
                            )
                            images_to_insert = set()
                            if key_frame:
                                norm_kf = self._normalize_image_path(key_frame)
                                if norm_kf:
                                    images_to_insert.add(norm_kf)
                            for f in saved_frames:
                                if f:
                                    norm_f = self._normalize_image_path(f)
                                    if norm_f:
                                        images_to_insert.add(norm_f)
                            if images_to_insert:
                                await db.executemany(
                                    "INSERT OR IGNORE INTO event_images (event_uid, image_path) VALUES (?, ?)",
                                    [(uid, img) for img in images_to_insert],
                                )
                        elif key_frame:
                            # Metadata-only update with key_frame: retain untouched existing frames, ensure key_frame is tracked
                            norm_kf = self._normalize_image_path(key_frame)
                            if norm_kf:
                                await db.execute(
                                    "INSERT OR IGNORE INTO event_images (event_uid, image_path) VALUES (?, ?)",
                                    (uid, norm_kf),
                                )
                        await db.commit()
                    else:
                        _LOGGER.warning(
                            f"Event with UID {uid} not found for update; skipping associations"
                        )
            await self.load_events()
        except aiosqlite.Error as e:
            _LOGGER.error(f"Error updating event in database: {e}")

    async def delete_event(
        self,
        uid: str,
    ) -> bool:
        """Deletes an event from the calendar and removes unshared assets safely."""
        await self._ensure_ready()
        _LOGGER.info(f"Deleting event with UID: {uid}")
        try:
            unshared_images_to_delete: list[str] = []
            async with self._get_mutation_lock():
                async with aiosqlite.connect(self._db_path) as db:
                    # 1. Fetch event key_frame
                    async with db.execute(
                        "SELECT key_frame FROM events WHERE uid = ?", (uid,)
                    ) as cur:
                        row = await cur.fetchone()
                        if not row:
                            return True
                        event_key_frame = row[0]

                    associated_images = set()
                    if event_key_frame:
                        associated_images.add(event_key_frame)

                    has_event_images = False
                    async with db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='event_images'"
                    ) as cur:
                        has_event_images = await cur.fetchone() is not None

                    if has_event_images:
                        async with db.execute(
                            "SELECT image_path FROM event_images WHERE event_uid = ?", (uid,)
                        ) as cur:
                            for img_row in await cur.fetchall():
                                if img_row[0]:
                                    associated_images.add(img_row[0])

                    # Fetch other events' keyframes and event_images for canonical ownership check
                    async with db.execute(
                        "SELECT key_frame FROM events WHERE uid != ? AND key_frame IS NOT NULL AND key_frame != ''",
                        (uid,),
                    ) as cur:
                        other_keyframes = {
                            self._normalize_image_path(r[0])
                            for r in await cur.fetchall()
                        }

                    other_saved_frames = set()
                    if has_event_images:
                        async with db.execute(
                            "SELECT image_path FROM event_images WHERE event_uid != ? AND image_path IS NOT NULL AND image_path != ''",
                            (uid,),
                        ) as cur:
                            other_saved_frames = {
                                self._normalize_image_path(r[0])
                                for r in await cur.fetchall()
                            }

                    other_all_frames = other_keyframes | other_saved_frames

                    for img in associated_images:
                        canon_img = self._normalize_image_path(img)
                        if canon_img and canon_img not in other_all_frames:
                            unshared_images_to_delete.append(img)

                    # Delete associations and event in the same transaction
                    if has_event_images:
                        await db.execute(
                            "DELETE FROM event_images WHERE event_uid = ?", (uid,)
                        )
                    await db.execute("DELETE FROM events WHERE uid = ?", (uid,))
                    await db.commit()

                # Filesystem deletion only happens after successful DB commit (still under lock)
                deleted_canonical = set()
                for img_path in unshared_images_to_delete:
                    canon = self._normalize_image_path(img_path)
                    if canon in deleted_canonical:
                        continue
                    if self._is_safe_snapshot_path(img_path):
                        try:
                            exists = await self.hass.async_add_executor_job(
                                os.path.exists, img_path
                            )
                            if exists:
                                await self.hass.async_add_executor_job(
                                    os.remove, img_path
                                )
                                deleted_canonical.add(canon)
                                _LOGGER.info(f"Deleted unshared image: {img_path}")
                        except OSError as e:
                            _LOGGER.warning(f"Failed to delete image {img_path}: {e}")

            await self.load_events()
            return True
        except Exception as e:
            _LOGGER.error(f"Error deleting event {uid}: {e}")
            return False

    async def _cleanup(self):
        """Deletes images not associated with any events.
        Protects:
          - Images linked to events.
          - Pending key_frames (event insert in progress).
          - Very new files (grace period).
        Fail-closed: If DB query fails, aborts immediately without removing any files.
        """
        await self._ensure_ready()
        if getattr(self, "_migrating", False):
            # Skip cleanup during migration
            return

        GRACE_SECONDS = 30

        # Run purge outside lock
        try:
            await self._purge_expired_events()
        except Exception as e:
            _LOGGER.warning(f"[CLEANUP] Pre-cleanup purge failed: {e}")

        async with self._get_mutation_lock():
            # Query linked frames directly from DB under lock (fail-closed)
            try:
                linked_frames = set()
                async with aiosqlite.connect(self._db_path) as db:
                    async with db.execute(
                        "SELECT key_frame FROM events WHERE key_frame IS NOT NULL AND key_frame != ''"
                    ) as cur:
                        for row in await cur.fetchall():
                            base = os.path.basename(row[0])
                            if base:
                                linked_frames.add(base.lower())

                    async with db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='event_images'"
                    ) as cur:
                        has_table = await cur.fetchone() is not None

                    if has_table:
                        async with db.execute(
                            "SELECT image_path FROM event_images WHERE image_path IS NOT NULL AND image_path != ''"
                        ) as cur:
                            for row in await cur.fetchall():
                                base = os.path.basename(row[0])
                                if base:
                                    linked_frames.add(base.lower())
            except Exception as e:
                _LOGGER.error(
                    f"[CLEANUP] Database read failed while querying linked frames: {e}. Aborting cleanup to fail closed."
                )
                return

            # List files in snapshots dir (in executor, non-blocking)
            try:
                files = await self.hass.async_add_executor_job(
                    os.listdir, self._media_path
                )
            except FileNotFoundError:
                return

            now_ts = datetime.datetime.now().timestamp()
            removed = 0

            for file in files:
                file_path = os.path.join(self._media_path, file)
                is_file = await self.hass.async_add_executor_job(
                    os.path.isfile, file_path
                )
                if not is_file:
                    continue

                base = (file or "").lower()

                # Protect if linked to an event or pending
                if base in linked_frames or base in self._pending_key_frames:
                    continue

                # Protect new files (grace window)
                try:
                    mtime = await self.hass.async_add_executor_job(
                        os.path.getmtime, file_path
                    )
                    if (now_ts - mtime) < GRACE_SECONDS:
                        continue
                except OSError:
                    continue

                _LOGGER.info(f"[CLEANUP] Removing unlinked snapshot: {file}")
                try:
                    await self.hass.async_add_executor_job(os.remove, file_path)
                    removed += 1
                except OSError as e:
                    _LOGGER.warning(f"[CLEANUP] Failed to remove {file_path}: {e}")

            if removed:
                _LOGGER.debug(f"[CLEANUP] Removed {removed} orphaned snapshot(s)")
