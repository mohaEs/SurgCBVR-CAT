"""Crash-resumable .npy feature matrices, with SQLite completion journal."""
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np

class FeatureCheckpoint:
    def __init__(self, files, paths, dim, settings):
        self.files = {k: Path(v) for k, v in files.items()}
        self.paths = [str(p) for p in paths]
        self.dim = int(dim)
        self.signature = hashlib.sha256(json.dumps({'paths': self.paths, 'settings': settings, 'dim': self.dim}, sort_keys=True).encode()).hexdigest()
        self.db_path = next(iter(self.files.values())).with_suffix('.progress.sqlite3')
        self.db = sqlite3.connect(self.db_path)
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS completed (idx INTEGER PRIMARY KEY, path TEXT NOT NULL, extra INTEGER)')
        old = self.db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if old and old[0] != self.signature:
            raise RuntimeError(f'Checkpoint settings or video list changed: {self.db_path}. Use a new output prefix or move the old outputs and progress DB.')
        if not old:
            if any(p.exists() for p in self.files.values()):
                raise RuntimeError('Feature file already exists without matching progress journal; move it aside before starting.')
            self.db.execute("INSERT INTO metadata VALUES ('signature', ?)", (self.signature,))
            self.db.commit()
        self.arrays = {}
        for key, path in self.files.items():
            if path.exists():
                a = np.load(path, mmap_mode='r+')
                if a.shape != (len(self.paths), self.dim) or a.dtype != np.float32:
                    raise RuntimeError(f'Unexpected shape or dtype in {path}')
            else:
                if old:
                    raise RuntimeError(f'Missing existing feature matrix {path}; cannot safely resume')
                a = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(len(self.paths), self.dim))
                a[:] = np.nan
                a.flush()
            self.arrays[key] = a
        self.done = {r[0] for r in self.db.execute('SELECT idx FROM completed')}
        print(f'Resuming: {len(self.done)}/{len(self.paths)} complete; journal: {self.db_path}', flush=True)

    def save(self, idx, vectors, extra=None):
        for key, vector in vectors.items():
            v = np.asarray(vector, dtype=np.float32)
            if v.shape != (self.dim,) or not np.isfinite(v).all():
                raise ValueError(f'Invalid feature for {key} at row {idx}: {v.shape}')
            self.arrays[key][idx] = v
        # Flush all arrays before committing completion; interrupted rows get recomputed.
        for a in self.arrays.values():
            a.flush()
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO completed VALUES (?, ?, ?)', (idx, self.paths[idx], extra))
        self.done.add(idx)

    def extras(self):
        return [r[0] for r in self.db.execute('SELECT extra FROM completed WHERE extra IS NOT NULL')]

    def close(self):
        self.db.close()
