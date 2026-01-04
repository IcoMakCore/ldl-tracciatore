#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Discord voice tracker (guild-only slash commands).
#
# Design notes (linux-ish style, adapted to python):
# - keep functions small and single-purpose
# - early returns, no deep nesting
# - explicit names, explicit data flow
# - avoid cleverness
#
# IMPORTANT:
# - Do NOT hardcode your token. Export DISCORD_TOKEN in the environment.
# - Commands are registered as GUILD-ONLY to appear immediately (no global delay).

import os
import io
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import aiosqlite
import discord
from discord.ext import commands

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DB_PATH = "voice_stats.sqlite3"

DEFAULT_TOP_N = 10
MAX_TOP_N = 25

DEFAULT_DAYS = 30
MAX_DAYS = 180

TZ = ZoneInfo("Europe/Rome")

# ---------------------------------------------------------------------
# GUILD-ONLY COMMANDS (Strada A)
# Set this to YOUR server id.
# ---------------------------------------------------------------------
GUILD_ID = 1227724065184415774
GUILD = discord.Object(id=GUILD_ID)


def now_ts() -> int:
    """Return current unix timestamp (seconds)."""
    return int(time.time())


def format_duration(seconds: int) -> str:
    """
    Format a duration in seconds to a short human-readable string.

    Example:
    - 3661 -> "1h 1m"
    - 61   -> "1m 1s"
    - 5    -> "5s"
    """
    seconds = max(0, int(seconds))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# Voice status priority:
# - Deaf wins over mute (if both true -> deaf)
STATUS_ACTIVE = 0
STATUS_MUTED = 1
STATUS_DEAF = 2


def voice_status(vs: discord.VoiceState | None) -> int:
    """
    Classify a voice state into one of:
    - active
    - muted
    - deaf

    Rules:
    - If self_deaf or server deaf -> STATUS_DEAF
    - Else if self_mute or server mute -> STATUS_MUTED
    - Else -> STATUS_ACTIVE
    """
    if vs is None:
        return STATUS_ACTIVE

    if bool(getattr(vs, "deaf", False)) or bool(getattr(vs, "self_deaf", False)):
        return STATUS_DEAF

    if bool(getattr(vs, "mute", False)) or bool(getattr(vs, "self_mute", False)):
        return STATUS_MUTED

    return STATUS_ACTIVE


def is_streaming(vs: discord.VoiceState | None) -> bool:
    """
    Return True if the member is currently streaming (screen share).

    discord.py uses 'self_stream' for users. Some versions expose 'streaming'.
    """
    if vs is None:
        return False

    return bool(getattr(vs, "self_stream", False)) or bool(getattr(vs, "streaming", False))


def split_segment_by_local_day(start_ts: int, end_ts: int):
    """
    Split a [start_ts, end_ts) interval into day-sized chunks (Europe/Rome).

    Yield tuples: (day_iso, seconds_in_that_day)

    Why:
    - daily graphs need stable day buckets
    - voice sessions can cross midnight
    """
    cur = int(start_ts)
    end_ts = int(end_ts)

    while cur < end_ts:
        cur_dt = datetime.fromtimestamp(cur, TZ)
        next_midnight = datetime.combine(cur_dt.date() + timedelta(days=1),
                                         dtime(0, 0),
                                         tzinfo=TZ)

        boundary = min(end_ts, int(next_midnight.timestamp()))
        yield cur_dt.date().isoformat(), boundary - cur
        cur = boundary


@dataclass
class UserTotals:
    """
    Accumulate voice usage totals.

    NOTE:
    - total voice time = active + muted + deaf
    - stream time is parallel, does not add to total voice time
    """
    active: int = 0
    muted: int = 0
    deaf: int = 0
    stream: int = 0

    @property
    def total(self) -> int:
        """Return total time spent in voice (excluding stream overlay)."""
        return self.active + self.muted + self.deaf

    def add_status(self, status: int, delta: int) -> None:
        """Add 'delta' seconds to the bucket chosen by 'status'."""
        delta = max(0, int(delta))

        if status == STATUS_ACTIVE:
            self.active += delta
            return
        if status == STATUS_MUTED:
            self.muted += delta
            return
        if status == STATUS_DEAF:
            self.deaf += delta
            return

    def add_stream(self, delta: int) -> None:
        """Add 'delta' seconds to streaming time."""
        self.stream += max(0, int(delta))

    def pie_values(self) -> list[int]:
        """Return values in [active, muted, deaf] order for the pie chart."""
        return [self.active, self.muted, self.deaf]


class VoiceTrackerDB:
    """
    SQLite persistence for:
    - total per user (voice_totals)
    - active session per user (voice_sessions)
    - daily buckets (voice_daily)

    Key invariants:
    - voice_sessions contains ONLY currently connected members
    - voice_sessions.last_ts is the last accounting point for status
    - voice_sessions.stream_last_ts is the last accounting point for streaming
    """

    def __init__(self, path: str):
        self.path = path
        self.db: aiosqlite.Connection | None = None

    async def _table_columns(self, table: str) -> set[str]:
        """
        Return set of column names for a given table.

        Used for soft migrations when older DB files miss new columns.
        """
        assert self.db is not None

        cur = await self.db.execute(f"PRAGMA table_info({table});")
        rows = await cur.fetchall()
        await cur.close()

        return {str(r[1]) for r in rows}

    async def _add_column_if_missing(self, table: str, coldef_sql: str, colname: str) -> None:
        """
        Add a column to an existing table if it does not exist.

        This allows you to keep your old DB file while evolving schema.
        """
        assert self.db is not None

        cols = await self._table_columns(table)
        if colname in cols:
            return

        await self.db.execute(f"ALTER TABLE {table} ADD COLUMN {coldef_sql};")
        await self.db.commit()

    async def connect(self) -> None:
        """
        Open the SQLite DB and ensure schema is present.

        WAL is enabled to reduce contention and improve write behavior.
        """
        self.db = await aiosqlite.connect(self.path)
        await self.db.execute("PRAGMA journal_mode=WAL;")

        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_totals (
                guild_id INTEGER NOT NULL,
                user_id  INTEGER NOT NULL,
                active_seconds INTEGER NOT NULL DEFAULT 0,
                muted_seconds  INTEGER NOT NULL DEFAULT 0,
                deaf_seconds   INTEGER NOT NULL DEFAULT 0,
                stream_seconds INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id)
            );
            """
        )

        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_sessions (
                guild_id INTEGER NOT NULL,
                user_id  INTEGER NOT NULL,
                join_ts  INTEGER NOT NULL,
                last_ts  INTEGER NOT NULL,
                status   INTEGER NOT NULL,
                stream_last_ts INTEGER NOT NULL,
                streaming INTEGER NOT NULL,
                PRIMARY KEY (guild_id, user_id)
            );
            """
        )

        await self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_daily (
                guild_id INTEGER NOT NULL,
                user_id  INTEGER NOT NULL,
                day      TEXT NOT NULL,
                active_seconds INTEGER NOT NULL DEFAULT 0,
                muted_seconds  INTEGER NOT NULL DEFAULT 0,
                deaf_seconds   INTEGER NOT NULL DEFAULT 0,
                stream_seconds INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, user_id, day)
            );
            """
        )

        await self.db.commit()

        # Soft migrations.
        await self._add_column_if_missing("voice_totals",
                                          "stream_seconds INTEGER NOT NULL DEFAULT 0",
                                          "stream_seconds")
        await self._add_column_if_missing("voice_daily",
                                          "stream_seconds INTEGER NOT NULL DEFAULT 0",
                                          "stream_seconds")

        await self._add_column_if_missing("voice_sessions",
                                          "last_ts INTEGER NOT NULL DEFAULT 0",
                                          "last_ts")
        await self._add_column_if_missing("voice_sessions",
                                          "status INTEGER NOT NULL DEFAULT 0",
                                          "status")
        await self._add_column_if_missing("voice_sessions",
                                          "stream_last_ts INTEGER NOT NULL DEFAULT 0",
                                          "stream_last_ts")
        await self._add_column_if_missing("voice_sessions",
                                          "streaming INTEGER NOT NULL DEFAULT 0",
                                          "streaming")

        # Normalize older rows (if any).
        await self.db.execute("UPDATE voice_sessions SET last_ts = join_ts WHERE last_ts = 0;")
        await self.db.execute("UPDATE voice_sessions SET stream_last_ts = last_ts WHERE stream_last_ts = 0;")
        await self.db.commit()

    async def close(self) -> None:
        """Close the DB connection."""
        if self.db is None:
            return

        await self.db.close()
        self.db = None

    async def start_session(self, guild_id: int, user_id: int, ts: int,
                            status: int, streaming: bool) -> None:
        """
        Start tracking a session if not already present.

        This is idempotent: repeated "start" calls do not create duplicates.
        """
        assert self.db is not None

        await self.db.execute(
            """
            INSERT INTO voice_sessions
                (guild_id, user_id, join_ts, last_ts, status, stream_last_ts, streaming)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(guild_id, user_id) DO NOTHING;
            """,
            (guild_id, user_id, ts, ts, int(status), ts, 1 if streaming else 0),
        )
        await self.db.commit()

    async def _add_status_segment(self, guild_id: int, user_id: int,
                                  status: int, start_ts: int, end_ts: int) -> None:
        """
        Account a time segment to totals and daily buckets for a given status.

        Called on:
        - status transition
        - session end
        """
        assert self.db is not None

        start_ts = int(start_ts)
        end_ts = int(end_ts)
        if end_ts <= start_ts:
            return

        delta = end_ts - start_ts
        a = delta if status == STATUS_ACTIVE else 0
        m = delta if status == STATUS_MUTED else 0
        d = delta if status == STATUS_DEAF else 0

        await self.db.execute(
            """
            INSERT INTO voice_totals
                (guild_id, user_id, active_seconds, muted_seconds, deaf_seconds, stream_seconds)
            VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT(guild_id, user_id) DO UPDATE SET
                active_seconds = active_seconds + excluded.active_seconds,
                muted_seconds  = muted_seconds  + excluded.muted_seconds,
                deaf_seconds   = deaf_seconds   + excluded.deaf_seconds;
            """,
            (guild_id, user_id, a, m, d),
        )

        for day_iso, secs in split_segment_by_local_day(start_ts, end_ts):
            a2 = secs if status == STATUS_ACTIVE else 0
            m2 = secs if status == STATUS_MUTED else 0
            d2 = secs if status == STATUS_DEAF else 0

            await self.db.execute(
                """
                INSERT INTO voice_daily
                    (guild_id, user_id, day, active_seconds, muted_seconds, deaf_seconds, stream_seconds)
                VALUES (?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(guild_id, user_id, day) DO UPDATE SET
                    active_seconds = active_seconds + excluded.active_seconds,
                    muted_seconds  = muted_seconds  + excluded.muted_seconds,
                    deaf_seconds   = deaf_seconds   + excluded.deaf_seconds;
                """,
                (guild_id, user_id, day_iso, a2, m2, d2),
            )

        await self.db.commit()

    async def _add_stream_segment(self, guild_id: int, user_id: int,
                                  start_ts: int, end_ts: int) -> None:
        """
        Account a time segment to streaming totals and daily buckets.

        Called on:
        - streaming transition (True -> False)
        - session end while streaming
        """
        assert self.db is not None

        start_ts = int(start_ts)
        end_ts = int(end_ts)
        if end_ts <= start_ts:
            return

        delta = end_ts - start_ts

        await self.db.execute(
            """
            INSERT INTO voice_totals
                (guild_id, user_id, active_seconds, muted_seconds, deaf_seconds, stream_seconds)
            VALUES (?, ?, 0, 0, 0, ?)
            ON CONFLICT(guild_id, user_id) DO UPDATE SET
                stream_seconds = stream_seconds + excluded.stream_seconds;
            """,
            (guild_id, user_id, delta),
        )

        for day_iso, secs in split_segment_by_local_day(start_ts, end_ts):
            await self.db.execute(
                """
                INSERT INTO voice_daily
                    (guild_id, user_id, day, active_seconds, muted_seconds, deaf_seconds, stream_seconds)
                VALUES (?, ?, ?, 0, 0, 0, ?)
                ON CONFLICT(guild_id, user_id, day) DO UPDATE SET
                    stream_seconds = stream_seconds + excluded.stream_seconds;
                """,
                (guild_id, user_id, day_iso, secs),
            )

        await self.db.commit()

    async def update_status_if_needed(self, guild_id: int, user_id: int,
                                      ts: int, new_status: int) -> None:
        """
        If the user is in-session and status has changed, account the previous
        segment and update the session status.
        """
        assert self.db is not None

        cur = await self.db.execute(
            "SELECT last_ts, status FROM voice_sessions WHERE guild_id=? AND user_id=?;",
            (guild_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()

        if not row:
            return

        last_ts = int(row[0])
        old_status = int(row[1])

        if new_status == old_status:
            return

        await self._add_status_segment(guild_id, user_id, old_status, last_ts, ts)

        await self.db.execute(
            "UPDATE voice_sessions SET last_ts=?, status=? WHERE guild_id=? AND user_id=?;",
            (ts, int(new_status), guild_id, user_id),
        )
        await self.db.commit()

    async def update_stream_if_needed(self, guild_id: int, user_id: int,
                                      ts: int, new_streaming: bool) -> None:
        """
        If the user is in-session and streaming state changed, account the
        previous streaming segment (if needed) and update streaming state.
        """
        assert self.db is not None

        cur = await self.db.execute(
            "SELECT stream_last_ts, streaming FROM voice_sessions WHERE guild_id=? AND user_id=?;",
            (guild_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()

        if not row:
            return

        stream_last_ts = int(row[0])
        old_streaming = int(row[1]) == 1
        new_streaming = bool(new_streaming)

        if new_streaming == old_streaming:
            return

        if old_streaming:
            await self._add_stream_segment(guild_id, user_id, stream_last_ts, ts)

        await self.db.execute(
            "UPDATE voice_sessions SET stream_last_ts=?, streaming=? WHERE guild_id=? AND user_id=?;",
            (ts, 1 if new_streaming else 0, guild_id, user_id),
        )
        await self.db.commit()

    async def end_session(self, guild_id: int, user_id: int, ts: int) -> None:
        """
        End a session and account remaining time (status + streaming if active).
        """
        assert self.db is not None

        cur = await self.db.execute(
            """
            SELECT last_ts, status, stream_last_ts, streaming
            FROM voice_sessions WHERE guild_id=? AND user_id=?;
            """,
            (guild_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()

        if not row:
            return

        last_ts = int(row[0])
        status = int(row[1])
        stream_last_ts = int(row[2])
        streaming = int(row[3]) == 1

        await self._add_status_segment(guild_id, user_id, status, last_ts, ts)

        if streaming:
            await self._add_stream_segment(guild_id, user_id, stream_last_ts, ts)

        await self.db.execute(
            "DELETE FROM voice_sessions WHERE guild_id=? AND user_id=?;",
            (guild_id, user_id),
        )
        await self.db.commit()

    async def get_sessions(self, guild_id: int) -> dict[int, tuple[int, int, int, int]]:
        """
        Return active sessions for a guild.

        Map:
            user_id -> (last_ts, status, stream_last_ts, streaming_int)
        """
        assert self.db is not None

        cur = await self.db.execute(
            """
            SELECT user_id, last_ts, status, stream_last_ts, streaming
            FROM voice_sessions WHERE guild_id=?;
            """,
            (guild_id,),
        )
        rows = await cur.fetchall()
        await cur.close()

        out: dict[int, tuple[int, int, int, int]] = {}
        for uid, last_ts, status, st_last, streaming in rows:
            out[int(uid)] = (int(last_ts), int(status), int(st_last), int(streaming))
        return out

    async def clear_sessions_not_in(self, guild_id: int, allowed_user_ids: set[int]) -> None:
        """
        On bot restart we cannot reconstruct offline time.

        This function removes stale sessions from DB for users who are NOT
        in voice anymore (according to current guild channel member list).
        """
        assert self.db is not None

        cur = await self.db.execute(
            "SELECT user_id FROM voice_sessions WHERE guild_id=?;",
            (guild_id,),
        )
        rows = await cur.fetchall()
        await cur.close()

        to_delete = [int(uid) for (uid,) in rows if int(uid) not in allowed_user_ids]
        if not to_delete:
            return

        await self.db.executemany(
            "DELETE FROM voice_sessions WHERE guild_id=? AND user_id=?;",
            [(guild_id, uid) for uid in to_delete],
        )
        await self.db.commit()

    async def get_totals_by_user(self, guild_id: int) -> dict[int, UserTotals]:
        """
        Return persisted totals for all users in a guild.
        """
        assert self.db is not None

        cur = await self.db.execute(
            """
            SELECT user_id, active_seconds, muted_seconds, deaf_seconds, stream_seconds
            FROM voice_totals WHERE guild_id=?;
            """,
            (guild_id,),
        )
        rows = await cur.fetchall()
        await cur.close()

        out: dict[int, UserTotals] = {}
        for uid, a, m, d, s in rows:
            out[int(uid)] = UserTotals(int(a), int(m), int(d), int(s))
        return out

    async def get_user_totals(self, guild_id: int, user_id: int) -> UserTotals:
        """
        Return persisted totals for a single user (no live time).
        """
        assert self.db is not None

        cur = await self.db.execute(
            """
            SELECT active_seconds, muted_seconds, deaf_seconds, stream_seconds
            FROM voice_totals WHERE guild_id=? AND user_id=?;
            """,
            (guild_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()

        if not row:
            return UserTotals()

        return UserTotals(int(row[0]), int(row[1]), int(row[2]), int(row[3]))

    async def get_user_daily(self, guild_id: int, user_id: int,
                             start_day_iso: str, end_day_iso: str) -> dict[str, UserTotals]:
        """
        Return persisted daily buckets for [start_day_iso, end_day_iso] inclusive.

        Map:
            day_iso -> UserTotals(active, muted, deaf, stream)
        """
        assert self.db is not None

        cur = await self.db.execute(
            """
            SELECT day, active_seconds, muted_seconds, deaf_seconds, stream_seconds
            FROM voice_daily
            WHERE guild_id=? AND user_id=? AND day >= ? AND day <= ?
            ORDER BY day ASC;
            """,
            (guild_id, user_id, start_day_iso, end_day_iso),
        )
        rows = await cur.fetchall()
        await cur.close()

        out: dict[str, UserTotals] = {}
        for day, a, m, d, s in rows:
            out[str(day)] = UserTotals(int(a), int(m), int(d), int(s))
        return out

    async def get_leaderboard_range(self, guild_id: int,
                                    start_day_iso: str, end_day_iso: str) -> dict[int, int]:
        """
        Return total voice seconds per user in a given day range (inclusive).

        This sums only voice time (active+muted+deaf). Streaming is excluded.
        """
        assert self.db is not None

        cur = await self.db.execute(
            """
            SELECT user_id, SUM(active_seconds + muted_seconds + deaf_seconds) AS total_sec
            FROM voice_daily
            WHERE guild_id=? AND day >= ? AND day <= ?
            GROUP BY user_id;
            """,
            (guild_id, start_day_iso, end_day_iso),
        )
        rows = await cur.fetchall()
        await cur.close()

        return {int(uid): int(total or 0) for uid, total in rows}


def make_pie_image(totals: UserTotals) -> io.BytesIO:
    """
    Generate a pie chart (active/muted/deaf) as a PNG in memory.
    """
    labels = ["Attivo", "Mutato", "Defenato"]
    values = totals.pie_values()

    # avoid matplotlib warnings on all-zero series
    if sum(values) <= 0:
        values = [1, 0, 0]

    fig = plt.figure()
    plt.pie(values, labels=labels, autopct="%1.1f%%")
    plt.title("Ripartizione tempo in vocale")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def make_line_image(days: list[str], values_hours: list[float], title: str) -> io.BytesIO:
    """
    Generate a line chart as a PNG in memory.

    days:
        list of labels (strings) for the x-axis
    values_hours:
        numeric series, same length as 'days'
    """
    fig = plt.figure()
    plt.plot(days, values_hours, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Ore")
    plt.title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


async def resolve_name(interaction: discord.Interaction, user_id: int) -> str:
    """
    Resolve a user id to a display name without pinging.

    Uses cache first, then fetches member if needed.
    """
    guild = interaction.guild
    if not guild:
        return str(user_id)

    m = guild.get_member(user_id)
    if m:
        return m.display_name

    try:
        m = await guild.fetch_member(user_id)
        return m.display_name
    except Exception:
        return f"User {user_id}"


intents = discord.Intents.default()
intents.voice_states = True

db = VoiceTrackerDB(DB_PATH)


async def reconcile_sessions_on_ready() -> None:
    """
    Reconcile DB sessions after a bot restart.

    What it does:
    - Detect users currently connected in voice/stage channels
    - Remove stale sessions from DB for users no longer in voice
    - For users in voice without an active DB session, start a session "from now"

    Limitation:
    - If the bot was offline, time during offline interval cannot be reconstructed.
    """
    for guild in bot.guilds:
        active_now: set[int] = set()

        channels = list(getattr(guild, "voice_channels", []))
        channels += list(getattr(guild, "stage_channels", []))

        for ch in channels:
            for member in ch.members:
                if member.bot:
                    continue
                active_now.add(member.id)

        await db.clear_sessions_not_in(guild.id, active_now)

        sessions = await db.get_sessions(guild.id)
        ts = now_ts()

        for uid in active_now:
            if uid in sessions:
                continue

            member = guild.get_member(uid)
            vs = member.voice if (member and member.voice) else None
            st = voice_status(vs)
            streaming = is_streaming(vs)
            await db.start_session(guild.id, uid, ts, st, streaming)


class VoiceBot(commands.Bot):
    async def setup_hook(self) -> None:
        """
        One-shot cleanup:
        1) remove ALL global commands from the application
        2) sync guild-only commands for immediate availability
        """
        await db.connect()

        # 1) Delete GLOBAL commands that may be lingering (old versions).
        self.tree.clear_commands(guild=None)
        g_synced = await self.tree.sync()
        print(f"[OK] Global cleared/synced: {len(g_synced)} -> {[c.name for c in g_synced]}")

        # 2) Sync GUILD-only commands (the ones with guild=GUILD in decorators).
        synced = await self.tree.sync(guild=GUILD)
        print(f"[OK] Guild synced: {len(synced)} -> {[c.name for c in synced]}")



# replace the bot instance with our custom one
bot = VoiceBot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    """Called when the websocket is ready and bot is logged in."""
    await reconcile_sessions_on_ready()
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")


@bot.event
async def on_voice_state_update(member: discord.Member,
                                before: discord.VoiceState,
                                after: discord.VoiceState):
    """
    Track voice usage.

    Events handled:
    - join voice: start session
    - leave voice: end session
    - stay in voice: account status/stream transitions (mute/deaf/stream toggles)

    Channel moves are treated as continuous (no close/reopen).
    """
    if member.bot:
        return

    guild_id = member.guild.id
    user_id = member.id
    ts = now_ts()

    before_ch = before.channel
    after_ch = after.channel

    if before_ch is None and after_ch is not None:
        await db.start_session(guild_id, user_id, ts,
                               voice_status(after),
                               is_streaming(after))
        return

    if before_ch is not None and after_ch is None:
        await db.end_session(guild_id, user_id, ts)
        return

    if before_ch is not None and after_ch is not None:
        await db.update_status_if_needed(guild_id, user_id, ts, voice_status(after))
        await db.update_stream_if_needed(guild_id, user_id, ts, is_streaming(after))
        return


# ---------------------------------------------------------------------
# GUILD-ONLY SLASH COMMANDS (Strada A): guild=GUILD
# ---------------------------------------------------------------------

@bot.tree.command(name="vocale_top",
                  description="Leaderboard totale: tempo passato in vocale (no ping).",
                  guild=GUILD)
async def vocale_top(interaction: discord.Interaction, top: int = DEFAULT_TOP_N):
    """
    Show total voice leaderboard for this server.

    Implementation:
    - take persisted totals from DB
    - add live session deltas (from last_ts to now)
    - sort by total voice time and show top N
    """
    if interaction.guild is None:
        await interaction.response.send_message("Questo comando funziona solo in un server.",
                                                ephemeral=True)
        return

    top = max(1, min(int(top), MAX_TOP_N))

    guild_id = interaction.guild.id
    ts = now_ts()

    totals_map = await db.get_totals_by_user(guild_id)
    sessions = await db.get_sessions(guild_id)

    combined = {uid: UserTotals(t.active, t.muted, t.deaf, t.stream)
                for uid, t in totals_map.items()}

    for uid, (last_ts, status, _st_last, _streaming) in sessions.items():
        delta = max(0, ts - last_ts)
        if uid not in combined:
            combined[uid] = UserTotals()
        combined[uid].add_status(status, delta)

    if not combined:
        await interaction.response.send_message("Non ho ancora dati sul tempo in vocale.")
        return

    ranking = sorted(combined.items(), key=lambda kv: kv[1].total, reverse=True)[:top]

    lines: list[str] = []
    for i, (uid, t) in enumerate(ranking, start=1):
        name = await resolve_name(interaction, uid)
        lines.append(f"{i:>2}. {name}  |  {format_duration(t.total)}")

    embed = discord.Embed(
        title="Leaderboard totale (vocale)",
        description="\n".join(lines),
        color=discord.Color.blurple(),
    )
    embed.set_footer(text="Include anche il tempo live corrente. Nessun ping.")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="stats",
                  description="Statistiche utente (totale/mutato/defenato/screenshare).",
                  guild=GUILD)
async def stats(interaction: discord.Interaction, user: discord.Member | None = None):
    """
    Show detailed stats for a user:
    - total voice time
    - active/muted/deaf breakdown
    - screen share time

    Includes live time for currently active session.
    """
    if interaction.guild is None:
        await interaction.response.send_message("Questo comando funziona solo in un server.",
                                                ephemeral=True)
        return

    user = user or interaction.user
    guild_id = interaction.guild.id
    ts = now_ts()

    totals = await db.get_user_totals(guild_id, user.id)
    sessions = await db.get_sessions(guild_id)

    if user.id in sessions:
        last_ts, status, stream_last_ts, streaming = sessions[user.id]
        totals.add_status(status, max(0, ts - last_ts))
        if int(streaming) == 1:
            totals.add_stream(max(0, ts - stream_last_ts))

    embed = discord.Embed(
        title=f"Statistiche vocali — {user.display_name}",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="Tempo totale in vocale", value=format_duration(totals.total), inline=False)
    embed.add_field(name="Attivo", value=format_duration(totals.active), inline=True)
    embed.add_field(name="Mutato", value=format_duration(totals.muted), inline=True)
    embed.add_field(name="Defenato", value=format_duration(totals.deaf), inline=True)
    embed.add_field(name="Screen share", value=format_duration(totals.stream), inline=True)

    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="vocale_giornaliera",
                  description="Leaderboard giornaliera: tempo in vocale di oggi (no ping).",
                  guild=GUILD)
async def vocale_giornaliera(interaction: discord.Interaction, top: int = DEFAULT_TOP_N):
    """
    Show today's voice leaderboard for the server.

    Implementation:
    - read today's bucket from voice_daily
    - add live time for currently connected members, clipped to today's day
    """
    if interaction.guild is None:
        await interaction.response.send_message("Questo comando funziona solo in un server.",
                                                ephemeral=True)
        return

    top = max(1, min(int(top), MAX_TOP_N))

    guild_id = interaction.guild.id
    ts = now_ts()

    today_iso = datetime.fromtimestamp(ts, TZ).date().isoformat()

    base = await db.get_leaderboard_range(guild_id, today_iso, today_iso)
    sessions = await db.get_sessions(guild_id)

    for uid, (last_ts, _status, _st_last, _streaming) in sessions.items():
        for day_iso, secs in split_segment_by_local_day(last_ts, ts):
            if day_iso == today_iso:
                base[uid] = base.get(uid, 0) + secs

    if not base:
        await interaction.response.send_message("Nessun dato per oggi (ancora).")
        return

    ranking = sorted(base.items(), key=lambda kv: kv[1], reverse=True)[:top]

    lines: list[str] = []
    for i, (uid, sec) in enumerate(ranking, start=1):
        name = await resolve_name(interaction, uid)
        lines.append(f"{i:>2}. {name}  |  {format_duration(sec)}")

    embed = discord.Embed(
        title="Leaderboard giornaliera (oggi)",
        description="\n".join(lines),
        color=discord.Color.blurple(),
    )
    embed.set_footer(text=f"Data: {today_iso} (Europe/Rome). Nessun ping.")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="vocale_settimanale",
                  description="Leaderboard settimanale: ultimi 7 giorni (no ping).",
                  guild=GUILD)
async def vocale_settimanale(interaction: discord.Interaction, top: int = DEFAULT_TOP_N):
    """
    Show last-7-days voice leaderboard for the server.

    Implementation:
    - read the range from voice_daily
    - add live time for current sessions, split per-day and included only if in range
    """
    if interaction.guild is None:
        await interaction.response.send_message("Questo comando funziona solo in un server.",
                                                ephemeral=True)
        return

    top = max(1, min(int(top), MAX_TOP_N))

    guild_id = interaction.guild.id
    ts = now_ts()

    end_day = datetime.fromtimestamp(ts, TZ).date()
    start_day = end_day - timedelta(days=6)

    start_iso = start_day.isoformat()
    end_iso = end_day.isoformat()

    base = await db.get_leaderboard_range(guild_id, start_iso, end_iso)
    sessions = await db.get_sessions(guild_id)

    for uid, (last_ts, _status, _st_last, _streaming) in sessions.items():
        for day_iso, secs in split_segment_by_local_day(last_ts, ts):
            if start_iso <= day_iso <= end_iso:
                base[uid] = base.get(uid, 0) + secs

    if not base:
        await interaction.response.send_message("Nessun dato per questa settimana (ancora).")
        return

    ranking = sorted(base.items(), key=lambda kv: kv[1], reverse=True)[:top]

    lines: list[str] = []
    for i, (uid, sec) in enumerate(ranking, start=1):
        name = await resolve_name(interaction, uid)
        lines.append(f"{i:>2}. {name}  |  {format_duration(sec)}")

    embed = discord.Embed(
        title="Leaderboard settimanale (ultimi 7 giorni)",
        description="\n".join(lines),
        color=discord.Color.blurple(),
    )
    embed.set_footer(text=f"Intervallo: {start_iso} → {end_iso} (Europe/Rome). Nessun ping.")
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="vocale_torta",
                  description="Grafico a torta: attivo / mutato / defenato per un utente.",
                  guild=GUILD)
async def vocale_torta(interaction: discord.Interaction, user: discord.Member | None = None):
    """
    Send a pie chart breakdown (active/muted/deaf) for a user.

    Includes live time from current session (status bucket only).
    """
    if interaction.guild is None:
        await interaction.response.send_message("Questo comando funziona solo in un server.",
                                                ephemeral=True)
        return

    user = user or interaction.user
    guild_id = interaction.guild.id
    ts = now_ts()

    totals = await db.get_user_totals(guild_id, user.id)
    sessions = await db.get_sessions(guild_id)

    if user.id in sessions:
        last_ts, status, _st_last, _streaming = sessions[user.id]
        totals.add_status(status, max(0, ts - last_ts))

    buf = make_pie_image(totals)
    file = discord.File(fp=buf, filename="vocale_torta.png")

    embed = discord.Embed(
        title=f"Ripartizione tempo in vocale: {user.display_name}",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="Attivo", value=format_duration(totals.active), inline=True)
    embed.add_field(name="Mutato", value=format_duration(totals.muted), inline=True)
    embed.add_field(name="Defenato", value=format_duration(totals.deaf), inline=True)
    embed.set_image(url="attachment://vocale_torta.png")

    await interaction.response.send_message(embed=embed, file=file)


@bot.tree.command(name="vocale_linea",
                  description="Grafico a linea: ore al giorno per un utente (totale/attivo/mutato/defenato).",
                  guild=GUILD)
async def vocale_linea(interaction: discord.Interaction,
                       user: discord.Member | None = None,
                       days: int = DEFAULT_DAYS,
                       mode: str = "totale"):
    """
    Send a line chart for a user's usage over N days.

    mode:
        - totale
        - attivo
        - mutato
        - defenato

    Includes live time (current session) distributed across day boundaries.
    """
    if interaction.guild is None:
        await interaction.response.send_message("Questo comando funziona solo in un server.",
                                                ephemeral=True)
        return

    user = user or interaction.user
    days = max(1, min(int(days), MAX_DAYS))

    mode = (mode or "totale").strip().lower()
    if mode not in ("totale", "attivo", "mutato", "defenato"):
        await interaction.response.send_message("mode deve essere: totale, attivo, mutato, defenato",
                                                ephemeral=True)
        return

    guild_id = interaction.guild.id
    ts = now_ts()

    end_day = datetime.fromtimestamp(ts, TZ).date()
    start_day = end_day - timedelta(days=days - 1)

    daily = await db.get_user_daily(guild_id, user.id,
                                    start_day.isoformat(),
                                    end_day.isoformat())

    sessions = await db.get_sessions(guild_id)
    if user.id in sessions:
        last_ts, status, _st_last, _streaming = sessions[user.id]
        if ts > last_ts:
            for day_iso, secs in split_segment_by_local_day(last_ts, ts):
                if day_iso not in daily:
                    daily[day_iso] = UserTotals()
                daily[day_iso].add_status(status, secs)

    labels: list[str] = []
    values_hours: list[float] = []

    cur = start_day
    while cur <= end_day:
        day_iso = cur.isoformat()
        t = daily.get(day_iso, UserTotals())

        if mode == "totale":
            secs = t.total
            title_mode = "Totale"
        elif mode == "attivo":
            secs = t.active
            title_mode = "Attivo"
        elif mode == "mutato":
            secs = t.muted
            title_mode = "Mutato"
        else:
            secs = t.deaf
            title_mode = "Defenato"

        labels.append(day_iso[5:])  # "MM-DD"
        values_hours.append(secs / 3600.0)
        cur += timedelta(days=1)

    buf = make_line_image(labels, values_hours,
                          f"{title_mode} per giorno: {user.display_name} ({days} giorni)")

    file = discord.File(fp=buf, filename="vocale_linea.png")

    embed = discord.Embed(
        title=f"Ore al giorno ({mode}) — {user.display_name}",
        description=f"Intervallo: {start_day.isoformat()} → {end_day.isoformat()} (TZ Europe/Rome)",
        color=discord.Color.blurple(),
    )
    embed.set_image(url="attachment://vocale_linea.png")

    await interaction.response.send_message(embed=embed, file=file)


def main() -> None:
    """
    Entry point.

    Requirements:
    - export DISCORD_TOKEN="..."
    - the bot must be invited with scope: bot + applications.commands
    """
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Missing DISCORD_TOKEN env var")


    bot.run(token)


if __name__ == "__main__":
    main()
