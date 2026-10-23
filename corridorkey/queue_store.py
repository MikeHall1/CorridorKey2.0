"""Single-owner durable queue observations; never an automatic resubmitter."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from .diagnostics import write_json_atomic
from .lifecycle import cleanup_all
from .local_json import read_object
from .queue_snapshot import capture_batch, validate_job
from .worker_protocol import (
    MAX_FRAME_BYTES,
    decode_frame,
    encode_frame,
    identifier,
    parse_request,
)

MAX_REQUESTS = 4096
TERMINAL = {'result', 'failed', 'cancelled', 'rejected', 'lost'}


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Queue file must be a regular owned file: {path.name}')


def _marker(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Choose an existing owned queue workspace.')
    path = root / 'queue-owner.json'
    _regular(path)
    with path.open('rb') as stream:
        raw = stream.read(MAX_FRAME_BYTES + 1)
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError('Queue ownership marker exceeds byte limit.')
    data = decode_frame(raw.rstrip(b'\n') + b'\n')
    if set(data) != {'schema_version', 'kind', 'queue_id'} or type(data['schema_version']) is not int or data['schema_version'] != 1 or data['kind'] != 'corridorkey_desktop_queue':
        raise ValueError('Not a supported CorridorKey queue workspace.')
    identifier(data['queue_id'])
    return data


def _text(value: dict[str, Any]) -> str:
    return encode_frame(value).decode().rstrip('\n')


def _archive(root: Path, queue_id: str) -> dict[str, Any] | None:
    path = root / 'queue-archive.json'
    if not path.exists() and not path.is_symlink():
        return None
    _regular(path)
    data, _ = read_object(path, max_bytes=4096, max_depth=8)
    if (set(data) != {'schema_version', 'kind', 'queue_id', 'created_utc'} or
            type(data['schema_version']) is not int or data['schema_version'] != 1 or
            data['kind'] != 'corridorkey_queue_archive' or data['queue_id'] != queue_id or
            not isinstance(data['created_utc'], str) or not 0 < len(data['created_utc']) <= 80):
        raise ValueError('Invalid queue archive marker; managed writes refused.')
    return data


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _relocations(root: Path, queue_id: str, archive: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """Validate imported history's separate, immutable batch location mapping."""
    path = root / 'queue-transfer.json'
    if not path.exists() and not path.is_symlink():
        return None
    _imported_path(root, directory=True)
    _regular(path)
    info = path.lstat()
    if info.st_nlink != 1 or getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
        raise ValueError('Queue relocation marker must be a single-link regular file.')
    data, _ = read_object(path, max_bytes=MAX_FRAME_BYTES, max_depth=16)
    payload = data.get('payload')
    if (archive is None or set(data) != {'payload', 'sha256'} or not isinstance(payload, dict) or
            data['sha256'] != _digest(_text(payload)) or
            set(payload) != {'schema_version', 'kind', 'queue_id', 'path_flavour', 'recoveries'} or
            type(payload.get('schema_version')) is not int or payload['schema_version'] != 1 or
            payload.get('kind') != 'corridorkey_queue_relocation' or payload.get('queue_id') != queue_id or
            payload.get('path_flavour') != ('windows' if os.name == 'nt' else 'posix') or
            not isinstance(payload.get('recoveries'), list) or len(payload['recoveries']) > MAX_REQUESTS):
        raise ValueError('Invalid or incompatible archived queue relocation marker.')
    result = {}
    for item in payload['recoveries']:
        if not isinstance(item, dict) or set(item) != {'request_id', 'historical_batch', 'local_batch', 'batch_sha256'}:
            raise ValueError('Invalid queue recovery relocation record.')
        request_id = identifier(item['request_id'])
        historical = item['historical_batch']
        if (request_id in result or not isinstance(historical, str) or len(historical) > 4096 or not Path(historical).is_absolute() or
                item['local_batch'] != f'batches/{request_id}.json' or
                not isinstance(item['batch_sha256'], str) or len(item['batch_sha256']) != 64 or
                any(character not in '0123456789abcdef' for character in item['batch_sha256'])):
            raise ValueError('Invalid queue batch relocation path/checksum.')
        result[request_id] = item
    return result


def _imported_path(path: Path, *, directory: bool = False) -> None:
    """Imported history admits regular local objects, including on Windows."""
    info = path.lstat()
    valid = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    if (not valid or (not directory and info.st_nlink != 1) or
            getattr(info, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)):
        raise ValueError('Imported queue paths must be regular owned objects without links/reparse points.')


class QueueStore:
    @classmethod
    def create(cls, parent: Path) -> Path:
        if not parent.is_dir():
            raise ValueError('Choose an existing queue parent directory.')
        root = parent.resolve() / f'queue-{uuid.uuid4().hex}'
        root.mkdir(exist_ok=False)
        queue_id = uuid.uuid4().hex
        write_json_atomic(root / 'queue-owner.json', {'schema_version': 1, 'kind': 'corridorkey_desktop_queue', 'queue_id': queue_id})
        with closing(sqlite3.connect(root / 'queue.sqlite')) as connection, connection:
            connection.executescript('''
                CREATE TABLE metadata(queue_id TEXT NOT NULL, schema_version INTEGER NOT NULL);
                CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_pid INTEGER, worker_pid INTEGER, generation TEXT,
                                      created REAL, closed INTEGER NOT NULL DEFAULT 0, clean INTEGER, error TEXT);
                CREATE TABLE requests(id TEXT PRIMARY KEY, session TEXT NOT NULL, action TEXT NOT NULL,
                                      request TEXT NOT NULL, request_hash TEXT NOT NULL, status TEXT NOT NULL,
                                      event TEXT, event_hash TEXT, created REAL, updated REAL, batch_hash TEXT);
            ''')
            connection.execute('INSERT INTO metadata VALUES (?,1)', (queue_id,))
        sqlite3.connect(root / 'owner-lock.sqlite').close()
        return root

    def __init__(self, root: Path):
        self.root = root.absolute()
        marker = _marker(self.root)
        for name in ('queue.sqlite', 'owner-lock.sqlite'):
            _regular(self.root / name)
        self.lock = sqlite3.connect(self.root / 'owner-lock.sqlite', timeout=0)
        self.db: sqlite3.Connection | None = None
        self.session: str | None = None
        self.progress_at: dict[str, float] = {}
        try:
            self.lock.execute('BEGIN EXCLUSIVE')
            if (self.root / 'queue-transfer.json').exists() or (self.root / 'queue-transfer.json').is_symlink():
                raise ValueError('Imported portable queue history is read-only; create a new queue for future jobs.')
            # Check after ownership acquisition: another owner may have sealed
            # history between our initial file checks and taking this lock.
            if _archive(self.root, marker['queue_id']) is not None:
                raise ValueError('Queue history is archived; create a new workspace for future jobs.')
            self.db = sqlite3.connect((self.root / 'queue.sqlite').as_uri() + '?mode=rw', uri=True, timeout=0)
            self.db.row_factory = sqlite3.Row
            self.db.execute('PRAGMA synchronous=FULL')
            rows = self.db.execute('SELECT queue_id,schema_version FROM metadata').fetchall()
            if len(rows) != 1 or rows[0]['queue_id'] != marker['queue_id'] or rows[0]['schema_version'] != 1:
                raise ValueError('Queue database ownership/version mismatch.')
            if 'batch_hash' not in {row[1] for row in self.db.execute('PRAGMA table_info(requests)')}:
                with self.db:
                    self.db.execute('ALTER TABLE requests ADD COLUMN batch_hash TEXT')
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        callbacks = [self.lock.close]
        if self.db is not None:
            callbacks.insert(0, self.db.close)
        cleanup_all(callbacks)

    @classmethod
    def seal(cls, root: Path) -> dict[str, Any]:
        # Exclusive managed ownership prevents racing a live journal writer.
        # Historical unresolved rows are retained, never relabelled completed.
        with closing(cls(root)):
            marker = _marker(root)
            data = {'schema_version': 1, 'kind': 'corridorkey_queue_archive', 'queue_id': marker['queue_id'],
                    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            stage = root / f'.archive-{uuid.uuid4().hex}.json'
            try:
                with stage.open('x', encoding='utf-8') as stream:
                    stream.write(_text(data) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                # Publish complete bytes without replacing any existing marker.
                os.link(stage, root / 'queue-archive.json')
            finally:
                stage.unlink(missing_ok=True)
        return data

    def start_session(self) -> str:
        assert self.db is not None
        if self.session is not None:
            raise ValueError('Queue owner already has a session.')
        session = uuid.uuid4().hex
        with self.db:
            self.db.execute('INSERT INTO sessions(id,parent_pid,created) VALUES (?,?,?)',
                            (session, os.getpid(), time.time()))
        self.session = session
        return self.session

    def connected(self, pid: int, generation: str) -> None:
        assert self.db is not None and self.session is not None
        with self.db:
            self.db.execute('UPDATE sessions SET worker_pid=?,generation=? WHERE id=?', (pid, generation, self.session))

    def finish(self, clean: bool, error: str | None) -> None:
        assert self.db is not None
        if self.session is not None:
            with self.db:
                self.db.execute('UPDATE sessions SET closed=1,clean=?,error=? WHERE id=?', (int(clean), error, self.session))

    def prepare(self, request_id: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.db is not None and self.session is not None
        request = parse_request({'schema_version': 1, 'request_id': request_id, 'action': action, 'payload': payload})
        if action not in {'render', 'recover'}:
            raise ValueError('Only render/recovery operations are durable queue requests.')
        if self.db.execute('SELECT COUNT(*) FROM requests').fetchone()[0] >= MAX_REQUESTS:
            raise ValueError('Durable queue history is full; create a new workspace.')
        payload = request.payload()
        batch_hash = None
        if action == 'recover':
            payload, batch_hash = capture_batch(self.root, request_id, payload)
        else:
            validate_job(self.root, payload['job'])
        text = _text({'schema_version': 1, 'request_id': request_id, 'action': action, 'payload': payload})
        with self.db:
            self.db.execute('INSERT INTO requests(id,session,action,request,request_hash,status,event,event_hash,created,updated,batch_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (request_id, self.session, action, text, _digest(text), 'prepared', None, None, time.time(), time.time(), batch_hash))
        return payload

    def phase(self, request_id: str, phase: str) -> None:
        assert self.db is not None
        if phase not in {'dispatching', 'sent'}:
            raise ValueError('Unsupported queue dispatch phase.')
        with self.db:
            previous = 'prepared' if phase == 'dispatching' else 'dispatching'
            changed = self.db.execute('UPDATE requests SET status=?,updated=? WHERE id=? AND session=? AND status=?',
                                      (phase, time.time(), request_id, self.session, previous)).rowcount
            if changed != 1:
                raise ValueError('Queue dispatch phase does not match current request state.')

    def observe(self, event: dict[str, Any]) -> None:
        assert self.db is not None
        request_id, kind = event.get('request_id'), event['kind']
        if not isinstance(request_id, str):
            return
        row = self.db.execute('SELECT status FROM requests WHERE id=? AND session=?', (request_id, self.session)).fetchone()
        if row is None or row['status'] in TERMINAL:
            return
        if kind not in {'accepted', 'started', 'progress', *TERMINAL}:
            return
        now = time.monotonic()
        if kind == 'progress' and now - self.progress_at.get(request_id, -1) < .25:
            return
        text = _text(event)
        with self.db:
            self.db.execute('UPDATE requests SET status=?,event=?,event_hash=?,updated=? WHERE id=?',
                            (kind, text, _digest(text), time.time(), request_id))
        if kind == 'progress':
            self.progress_at[request_id] = now

    @staticmethod
    def inspect(root: Path, *, request_id: str | None = None, before: int | None = None) -> dict[str, Any]:
        if before is not None and (type(before) is not int or not 0 < before < 2**63):
            raise ValueError('History cursor must be a positive SQLite sequence integer.')
        if before is not None and request_id is not None:
            raise ValueError('A history cursor cannot select a request detail.')
        marker = _marker(root)
        archive = _archive(root, marker['queue_id'])
        relocations = _relocations(root, marker['queue_id'], archive)
        _regular(root / 'queue.sqlite')
        with closing(sqlite3.connect((root.absolute() / 'queue.sqlite').as_uri() + '?mode=ro', uri=True, timeout=0)) as db:
            db.row_factory = sqlite3.Row
            metadata = db.execute('SELECT queue_id,schema_version FROM metadata').fetchall()
            if len(metadata) != 1 or metadata[0]['queue_id'] != marker['queue_id'] or metadata[0]['schema_version'] != 1:
                raise ValueError('Queue database ownership/version mismatch.')
            if request_id is None:
                # Admission sequence avoids duplicates/skips when timestamps tie
                # or a writer adds newer observations while history is paged.
                rows = db.execute('''SELECT r.rowid AS ordinal,r.id,r.action,r.status,r.session,r.created,s.worker_pid,s.generation,s.closed,s.clean
                                     FROM requests r JOIN sessions s ON r.session=s.id
                                     WHERE r.rowid < ? ORDER BY r.rowid DESC LIMIT 129''',
                                  (before if before is not None else 2**63 - 1,)).fetchall()
                next_before = rows[127]['ordinal'] if len(rows) > 128 else None
                headers = [dict(row) for row in rows[:128]]
                for header in headers:
                    header.pop('ordinal')
                return {'queue_id': marker['queue_id'], 'archive': archive, 'rows': headers, 'before': before, 'next_before': next_before,
                        'scope': 'Saved observations only. Old PIDs/process state and outputs NOT VERIFIED; no work resubmitted.'}
            row = db.execute('SELECT * FROM requests WHERE id=?', (identifier(request_id),)).fetchone()
            if row is None:
                raise ValueError('No such queue request.')
            record = dict(row)
            for name in ('request', 'event'):
                text = record[name]
                if text is not None:
                    if len(text.encode()) > MAX_FRAME_BYTES or _digest(text) != record[name + '_hash']:
                        raise ValueError('Queue observation content checksum/size mismatch.')
                    record[name] = decode_frame(text.encode() + b'\n')
            request = parse_request(record['request'])
            if request.request_id != record['id'] or request.action != record['action']:
                raise ValueError('Queue request metadata differs from its saved envelope.')
            if record.get('batch_hash') is not None:
                relocation = relocations.get(record['id']) if relocations is not None else None
                if relocations is not None and (relocation is None or request.action != 'recover' or
                        relocation['historical_batch'] != request.payload()['batch'] or relocation['batch_sha256'] != record['batch_hash']):
                    raise ValueError('Queue relocation differs from original checksummed request.')
                batch = root / relocation['local_batch'] if relocation is not None else Path(request.payload()['batch'])
                if relocation is not None:
                    _imported_path(root / 'batches', directory=True)
                    _imported_path(batch)
                if (root / 'batches').is_symlink() or (root / 'batches').is_junction():
                    raise ValueError('Queue batch snapshot directory must be owned.')
                if not batch.resolve().is_relative_to((root / 'batches').resolve()):
                    raise ValueError('Queue batch snapshot is outside its owned directory.')
                _regular(batch)
                with batch.open('rb') as stream:
                    raw = stream.read(MAX_FRAME_BYTES + 1)
                if len(raw) > MAX_FRAME_BYTES or hashlib.sha256(raw).hexdigest() != record['batch_hash']:
                    raise ValueError('Queue batch snapshot checksum/size mismatch.')
                if relocation is not None:
                    if batch.lstat().st_nlink != 1:
                        raise ValueError('Queue batch snapshot cannot be hardlinked.')
                    record['load_payload'] = {**request.payload(), 'batch': str(batch.absolute())}
            elif relocations is not None and request.action == 'recover':
                raise ValueError('Imported recovery history requires an original batch checksum.')
            event = record['event']
            if event is not None:
                if event.get('request_id') != record['id'] or event.get('kind') != record['status']:
                    raise ValueError('Queue event metadata differs from its saved envelope.')
            elif record['status'] not in {'prepared', 'dispatching', 'sent'}:
                raise ValueError('Queue terminal state has no saved observation.')
            record['scope'] = 'Historical observations, not output validation or live-process authority.'
            record['archive'] = archive
            return record
