CREATE TABLE task_runs (
  id                   TEXT PRIMARY KEY,
  task_id              TEXT,
  run_family_id        TEXT NOT NULL,
  execution_mode       TEXT NOT NULL,
  launch_profile       TEXT NOT NULL,
  review_status_id     TEXT NOT NULL,
  completion_status_id TEXT NOT NULL,
  merge_target         TEXT,
  state                TEXT NOT NULL,
  workspace_id         TEXT NOT NULL,
  source               TEXT NOT NULL,
  created_at           TEXT NOT NULL,
  completed_at         TEXT,
  backgrounded         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE task_run_segments (
  id                 TEXT PRIMARY KEY,
  task_run_id        TEXT NOT NULL,
  pane_id            TEXT NOT NULL,
  workspace_id       TEXT NOT NULL,
  adapter_session_id TEXT,
  state              TEXT NOT NULL,
  started_at         TEXT NOT NULL,
  ended_at           TEXT,
  ended_reason       TEXT,
  exit_code          INTEGER
);

INSERT INTO task_runs VALUES
  ('task-run-synthetic-alpha', NULL, 'run-family-synthetic-alpha', 'invented-mode',
   'invented-profile', 'invented-review', 'invented-completion', NULL, 'closed',
   'workspace-synthetic-alpha', 'invented-source', '2026-08-14T09:59:00Z',
   '2026-08-14T10:03:00Z', 0);

INSERT INTO task_run_segments VALUES
  ('segment-synthetic-alpha', 'task-run-synthetic-alpha', 'pane-synthetic-beta',
   'workspace-synthetic-alpha', 'session-synthetic-beta', 'interrupted',
   '2026-08-14T09:59:00Z', '2026-08-14T10:03:00Z', 'invented-end', 9);

INSERT INTO task_run_segments VALUES
  ('segment-synthetic-no-session', 'task-run-synthetic-alpha', 'pane-synthetic-alpha',
   'workspace-synthetic-alpha', NULL, 'interrupted',
   '2026-08-14T10:00:30Z', '2026-08-14T10:00:45Z',
   'PRIVATE SYNTHETIC OMITTED SEGMENT', 9);
