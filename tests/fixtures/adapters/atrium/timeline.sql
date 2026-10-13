CREATE TABLE timeline (
  id            TEXT PRIMARY KEY,
  workspace_id  TEXT NOT NULL,
  kind          TEXT NOT NULL,
  scope         TEXT NOT NULL,
  title         TEXT,
  body          TEXT,
  metadata_json TEXT,
  tags_json     TEXT,
  pane_id       TEXT,
  created_at    TEXT NOT NULL,
  backfill_key  TEXT,
  actor         TEXT
);

INSERT INTO timeline VALUES
  ('timeline-synthetic-session', 'workspace-synthetic-alpha', 'session-end', 'workspace',
   'PRIVATE SYNTHETIC TITLE', 'PRIVATE SYNTHETIC BODY',
   '{"adapter":"invented-harness","sessionId":"session-synthetic-alpha"}',
   '[]', 'pane-synthetic-alpha', '2026-08-14T10:00:00Z', NULL,
   'PRIVATE SYNTHETIC ACTOR'),
  ('timeline-synthetic-message', 'workspace-synthetic-alpha', 'agent-message', 'workspace',
   'PRIVATE SYNTHETIC MESSAGE TITLE', 'PRIVATE SYNTHETIC MESSAGE BODY',
   '{"fromPaneId":"pane-synthetic-alpha","targetPaneId":"pane-synthetic-beta","message":"PRIVATE SYNTHETIC MESSAGE"}',
   '[]', 'pane-synthetic-alpha', '2026-08-14T10:01:00Z', NULL,
   'PRIVATE SYNTHETIC MESSAGE ACTOR'),
  ('timeline-synthetic-subagent', 'workspace-synthetic-alpha', 'subagent-stop', 'workspace',
   'PRIVATE SYNTHETIC SUBAGENT TITLE', 'PRIVATE SYNTHETIC SUBAGENT BODY',
   '{"adapter":"invented-harness","subagent":""}',
   '[]', 'pane-synthetic-alpha', '2026-08-14T10:02:00Z', NULL,
   'PRIVATE SYNTHETIC SUBAGENT ACTOR');

INSERT INTO timeline VALUES
  ('timeline-synthetic-unsupported', 'workspace-synthetic-alpha',
   'PRIVATE SYNTHETIC UNSUPPORTED KIND', 'workspace',
   'PRIVATE SYNTHETIC OMITTED TITLE', 'PRIVATE SYNTHETIC OMITTED BODY',
   '{"message":"PRIVATE SYNTHETIC OMITTED MESSAGE"}',
   '[]', 'pane-synthetic-alpha', '2026-08-14T10:01:30Z', NULL,
   'PRIVATE SYNTHETIC OMITTED ACTOR');
