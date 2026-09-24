PRAGMA foreign_keys = ON;

CREATE TABLE projects (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE hosts (
    id TEXT PRIMARY KEY NOT NULL
);

CREATE TABLE environments (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL,
    host_id TEXT NOT NULL,
    path TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (host_id) REFERENCES hosts(id) ON DELETE CASCADE
);

CREATE TABLE threads (
    id TEXT PRIMARY KEY NOT NULL,
    project_id TEXT NOT NULL,
    environment_id TEXT,
    provider_id TEXT NOT NULL,
    model_override TEXT,
    reasoning_level_override TEXT,
    title TEXT,
    title_fallback TEXT,
    parent_thread_id TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    source_thread_id TEXT REFERENCES threads(id) ON DELETE SET NULL,
    origin_kind TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (environment_id) REFERENCES environments(id) ON DELETE SET NULL,
    FOREIGN KEY (parent_thread_id) REFERENCES threads(id) ON DELETE SET NULL
);

CREATE INDEX threads_parent_idx ON threads(parent_thread_id);
CREATE INDEX threads_source_origin_idx ON threads(source_thread_id, origin_kind);

CREATE TABLE events (
    id TEXT PRIMARY KEY NOT NULL,
    thread_id TEXT NOT NULL,
    environment_id TEXT,
    scope_kind TEXT NOT NULL,
    turn_id TEXT,
    provider_thread_id TEXT,
    sequence INTEGER NOT NULL,
    type TEXT NOT NULL,
    data TEXT DEFAULT '{}' NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE,
    FOREIGN KEY (environment_id) REFERENCES environments(id) ON DELETE SET NULL,
    CONSTRAINT events_scope_shape_check CHECK (
        (scope_kind = 'turn' AND turn_id IS NOT NULL)
        OR (scope_kind = 'thread' AND turn_id IS NULL)
    )
);

CREATE UNIQUE INDEX events_thread_sequence_idx ON events(thread_id, sequence);

CREATE TABLE queued_thread_messages (
    id TEXT PRIMARY KEY NOT NULL,
    thread_id TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    reasoning_level TEXT NOT NULL,
    permission_mode TEXT NOT NULL,
    service_tier TEXT NOT NULL,
    sort_key TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    sender_thread_id TEXT,
    FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

INSERT INTO projects VALUES
    ('project_synth', 'Synthetic Project', 1788278400000, 1788278460000);
INSERT INTO hosts VALUES ('host_synth');
INSERT INTO environments VALUES
    ('env_synth', 'project_synth', 'host_synth', '/synthetic/workspace',
     1788278400000, 1788278460000);

INSERT INTO threads (
    id, project_id, environment_id, provider_id, model_override,
    reasoning_level_override, title, title_fallback, parent_thread_id,
    created_at, updated_at, source_thread_id, origin_kind
) VALUES
    ('thread_root_01', 'project_synth', 'env_synth', 'codex', NULL, NULL,
     'Private synthetic root title', 'Private fallback', NULL,
     1788278400000, 1788278460000, NULL, NULL),
    ('thread_child01', 'project_synth', 'env_synth', 'claude-code', NULL, NULL,
     'Private synthetic child title', 'Private fallback', 'thread_root_01',
     1788278520000, 1788278640000, NULL, NULL),
    ('thread_fork_01', 'project_synth', 'env_synth', 'codex', NULL, NULL,
     'Private synthetic fork title', 'Private fallback', NULL,
     1788278700000, 1788278880000, 'thread_root_01', 'fork');

-- Root: one sparse event, then legacy JSON fallback.
INSERT INTO events VALUES
    ('event_root_sparse', 'thread_root_01', 'env_synth', 'thread', NULL, NULL,
     1, 'client/thread/start', '{"private":"Private synthetic event data"}',
     1788278400000),
    ('event_root_identity', 'thread_root_01', 'env_synth', 'thread', NULL, NULL,
     2, 'thread/identity',
     '{"providerThreadId":"018f0000-0000-7000-8000-000000000001"}',
     1788278401000),
    ('event_root_tokens', 'thread_root_01', 'env_synth', 'thread', NULL, NULL,
     3, 'thread/tokenUsage/updated',
     '{"providerThreadId":"018f0000-0000-7000-8000-000000000001","tokenUsage":{"total":{"inputTokens":987654321,"cachedInputTokens":876543210,"outputTokens":765432109,"reasoningOutputTokens":654321098,"totalTokens":999999999}}}',
     1788278459000);

-- Child: provider UUID exists only in the first-class column.
INSERT INTO events VALUES
    ('event_child_sparse', 'thread_child01', 'env_synth', 'thread', NULL, NULL,
     1, 'system/thread-provisioning', '{}', 1788278520000),
    ('event_child_identity', 'thread_child01', 'env_synth', 'thread', NULL,
     '11111111-1111-4111-8111-111111111111', 2, 'thread/identity', '{}',
     1788278521000),
    ('event_child_tokens', 'thread_child01', 'env_synth', 'thread', NULL,
     '11111111-1111-4111-8111-111111111111', 3,
     'thread/tokenUsage/updated',
     '{"tokenUsage":{"total":{"inputTokens":912345678,"cachedInputTokens":812345678,"outputTokens":712345678,"reasoningOutputTokens":612345678,"totalTokens":999999998}}}',
     1788278639000);

-- Fork: current column and legacy JSON agree.
INSERT INTO events VALUES
    ('event_fork_sparse', 'thread_fork_01', 'env_synth', 'thread', NULL, NULL,
     1, 'system/operation', '{}', 1788278700000),
    ('event_fork_identity', 'thread_fork_01', 'env_synth', 'thread', NULL,
     '018f0000-0000-7000-8000-000000000002', 2, 'thread/identity',
     '{"providerThreadId":"018f0000-0000-7000-8000-000000000002"}',
     1788278701000),
    ('event_fork_tokens', 'thread_fork_01', 'env_synth', 'thread', NULL,
     '018f0000-0000-7000-8000-000000000002', 3,
     'thread/tokenUsage/updated',
     '{"providerThreadId":"018f0000-0000-7000-8000-000000000002","tokenUsage":{"total":{"inputTokens":923456789,"cachedInputTokens":823456789,"outputTokens":723456789,"reasoningOutputTokens":623456789,"totalTokens":999999997}}}',
     1788278879000);
