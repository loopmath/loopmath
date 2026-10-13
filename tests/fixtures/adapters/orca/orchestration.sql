PRAGMA foreign_keys = ON;

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_id TEXT,
    spec TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'pending', 'ready', 'dispatched', 'completed', 'failed', 'blocked'
    )),
    created_at TEXT NOT NULL,
    FOREIGN KEY(parent_id) REFERENCES tasks(id)
);

CREATE TABLE dispatch_contexts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'pending', 'dispatched', 'completed', 'failed', 'circuit_broken'
    )),
    created_at TEXT NOT NULL,
    FOREIGN KEY(task_id) REFERENCES tasks(id)
);

CREATE TABLE worker_dispatches (
    dispatch_id TEXT PRIMARY KEY,
    state TEXT NOT NULL CHECK(state IN (
        'starting', 'ready', 'start_unknown', 'failed', 'succeeded',
        'stopping', 'stop_unknown', 'stopped', 'abandoned'
    )),
    stage TEXT NOT NULL,
    worktree_id TEXT,
    start_options TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(dispatch_id) REFERENCES dispatch_contexts(id)
);

CREATE TABLE messages (
    id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    from_handle TEXT NOT NULL,
    to_handle TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL CHECK(type IN (
        'status', 'dispatch', 'worker_done', 'merge_ready', 'escalation',
        'handoff', 'decision_gate', 'question', 'heartbeat'
    )),
    payload TEXT,
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL
);

INSERT INTO runs VALUES
    ('run-synthetic-alpha', 'Invented alpha objective', '2026-08-10 10:00:00', '2026-08-10 10:20:00'),
    ('run-synthetic-beta', 'Invented beta objective', '2026-08-11 09:00:00', '2026-08-11 09:05:00');

INSERT INTO tasks VALUES
    ('task-alpha-plan', 'run-synthetic-alpha', NULL, 'Invented planning task', 'completed', '2026-08-10 10:00:01'),
    ('task-alpha-build', 'run-synthetic-alpha', 'task-alpha-plan', 'Invented build task', 'completed', '2026-08-10 10:02:00'),
    ('task-alpha-review', 'run-synthetic-alpha', 'task-alpha-build', 'Invented review task', 'completed', '2026-08-10 10:08:00'),
    ('task-alpha-gate', 'run-synthetic-alpha', 'task-alpha-review', 'Invented gate task', 'blocked', '2026-08-10 10:15:00'),
    ('task-beta-solo', 'run-synthetic-beta', NULL, 'Invented independent task', 'completed', '2026-08-11 09:00:01');

INSERT INTO dispatch_contexts VALUES
    ('dispatch-alpha-plan', 'run-synthetic-alpha', 'task-alpha-plan', 'completed', '2026-08-10 10:00:02'),
    ('dispatch-alpha-build', 'run-synthetic-alpha', 'task-alpha-build', 'completed', '2026-08-10 10:02:01'),
    ('dispatch-alpha-review', 'run-synthetic-alpha', 'task-alpha-review', 'completed', '2026-08-10 10:08:01'),
    ('dispatch-alpha-gate', 'run-synthetic-alpha', 'task-alpha-gate', 'dispatched', '2026-08-10 10:15:01'),
    ('dispatch-beta-solo', 'run-synthetic-beta', 'task-beta-solo', 'completed', '2026-08-11 09:00:02');

INSERT INTO worker_dispatches VALUES
    ('dispatch-alpha-plan', 'succeeded', 'settled', 'worktree-synthetic-alpha', '{"agent":"codex","launch":{"requested":{"agent":"codex","model":"gpt-5.6-sol","effort":"xhigh"},"effective":{"agent":"codex","model":"gpt-5.6-sol","effort":"xhigh"}}}', '2026-08-10 10:00:03', '2026-08-10 10:02:00'),
    ('dispatch-alpha-build', 'succeeded', 'settled', 'worktree-synthetic-alpha', '{"agent":"claude","launch":{"requested":{"agent":"claude","model":"claude-opus-5","effort":"max"},"effective":{"agent":"claude","model":"claude-opus-5","effort":"max"}}}', '2026-08-10 10:02:02', '2026-08-10 10:08:00'),
    ('dispatch-alpha-review', 'succeeded', 'settled', 'worktree-synthetic-alpha', '{"agent":"codex","launch":{"requested":{"agent":"codex","model":"gpt-5.6-sol","effort":"high"},"effective":{"agent":"codex","model":"gpt-5.6-sol","effort":"high"}}}', '2026-08-10 10:08:02', '2026-08-10 10:15:00'),
    ('dispatch-alpha-gate', 'ready', 'accepted', 'worktree-synthetic-alpha', 'PRIVATE SYNTHETIC MALFORMED START OPTIONS {', '2026-08-10 10:15:02', '2026-08-10 10:15:03'),
    ('dispatch-beta-solo', 'succeeded', 'settled', 'worktree-synthetic-beta', '{"agent":"codex","launch":{"requested":{"agent":"codex","model":"gpt-5.6-sol","effort":"medium"},"effective":{"agent":"codex","model":"gpt-5.6-sol","effort":"medium"}}}', '2026-08-11 09:00:03', '2026-08-11 09:05:00');

INSERT INTO messages (id, run_id, from_handle, to_handle, subject, body, type, payload, created_at) VALUES
    ('message-alpha-escalation', 'run-synthetic-alpha', 'worker-alpha', 'coordinator-alpha', 'Invented escalation', 'Invented body', 'escalation', '{"taskId":"task-alpha-build","dispatchId":"dispatch-alpha-build"}', '2026-08-10 10:07:00'),
    ('message-alpha-done', 'run-synthetic-alpha', 'worker-alpha', 'coordinator-alpha', 'Invented completion', 'Invented body', 'worker_done', '{"taskId":"task-alpha-build","dispatchId":"dispatch-alpha-build","outcome":"succeeded"}', '2026-08-10 10:08:00'),
    ('message-alpha-ready', 'run-synthetic-alpha', 'reviewer-alpha', 'coordinator-alpha', 'Invented readiness', 'Invented body', 'merge_ready', '{"taskId":"task-alpha-review","dispatchId":"dispatch-alpha-review"}', '2026-08-10 10:14:00'),
    ('message-alpha-gate', 'run-synthetic-alpha', 'coordinator-alpha', 'worker-alpha', 'Invented decision gate', 'Invented body', 'decision_gate', 'PRIVATE SYNTHETIC MALFORMED MESSAGE PAYLOAD {', '2026-08-10 10:16:00'),
    ('message-alpha-status', 'run-synthetic-alpha', 'worker-alpha', 'coordinator-alpha', 'Invented status', 'Invented body', 'status', '{"taskId":"task-alpha-plan","dispatchId":"dispatch-alpha-plan"}', '2026-08-10 10:01:00'),
    ('message-beta-status', 'run-synthetic-beta', 'worker-beta', 'coordinator-beta', 'Invented status', 'Invented body', 'status', '{"taskId":"task-beta-solo","dispatchId":"dispatch-beta-solo"}', '2026-08-11 09:01:00');
