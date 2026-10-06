-- ============================================================================
--  query_log — one row per question that passes through the system.
--
--  This table IS the deliverable. Every number in the README and on the resume
--  is a query over it. Columns for layers that don't exist yet are declared now
--  and left NULL, so there is never a mid-project migration.
--
--  Conventions:
--    * timestamps  : ISO-8601 UTC strings ('2026-09-12T18:04:11Z')
--    * lists/dicts : JSON in a TEXT column (don't normalise — this table tops
--                    out in the low thousands of rows)
--    * booleans    : INTEGER 0/1
--    * money       : REAL, USD
--    * NULL        : "this layer didn't run", never "it ran and found nothing"
-- ============================================================================

CREATE TABLE IF NOT EXISTS query_log (

  -- ==========================================================================
  -- IDENTITY
  -- run_id groups everything executed together; question_id is stable for the
  -- SAME question across runs. That pair is what lets you compare two configs
  -- later without re-labelling anything.
  -- ==========================================================================
  query_id            TEXT PRIMARY KEY,
  run_id              TEXT NOT NULL,
  question_id         TEXT NOT NULL,
  pipeline_config_id  TEXT NOT NULL,
  created_at          TEXT NOT NULL,

  -- shadow_mode = 1 means every layer ran regardless of early termination.
  -- Dev/eval runs: 1. Deployed demo: 0. Without this flag you cannot tell a
  -- "cascade stopped at L1" row apart from a "L2/L3 were never wired up" row.
  shadow_mode         INTEGER NOT NULL DEFAULT 0 CHECK (shadow_mode IN (0,1)),

  -- ==========================================================================
  -- INPUT
  -- induction_mode records HOW a wrong answer was manufactured. It is the
  -- column that produces the per-mode breakdown in the README — i.e. the
  -- honest finding about which failures the cheap layers cannot see.
  -- ==========================================================================
  query_text          TEXT NOT NULL,
  source_contract_id  TEXT,
  gold_chunk_ids      TEXT,              -- JSON array; which chunk(s) SHOULD have been retrieved
  induction_mode      TEXT CHECK (induction_mode IN (
                        'natural',
                        'unanswerable',
                        'starved_retrieval',
                        'weak_generator',
                        'distractor'
                      )),

  -- ==========================================================================
  -- RETRIEVAL
  -- retrieved_scores holds ALL k scores, not just the max. The distribution
  -- shape ("one strong match" vs "five mediocre ones") is a Week-4 classifier
  -- feature, and it is unrecoverable without re-running everything.
  -- ==========================================================================
  embedding_model     TEXT NOT NULL,
  retrieval_k         INTEGER NOT NULL,
  retrieved_chunk_ids TEXT NOT NULL,     -- JSON array, rank order
  retrieved_scores    TEXT NOT NULL,     -- JSON array, same order, ALL k
  retrieved_text      TEXT NOT NULL,     -- the context actually sent to the generator
  retrieval_hit       INTEGER CHECK (retrieval_hit IN (0,1)),  -- gold chunk present? NULL if no gold

  -- ==========================================================================
  -- GENERATION
  -- ==========================================================================
  generator_model     TEXT NOT NULL,
  generation_params   TEXT,              -- JSON: temperature, max_tokens, …
  prompt_version      TEXT,              -- bump when the RAG prompt changes
  answer_text         TEXT NOT NULL,
  prompt_tokens       INTEGER,
  completion_tokens   INTEGER,

  -- ==========================================================================
  -- LAYER 0 — input guardrails (runs before retrieval)
  -- ==========================================================================
  l0_flagged          INTEGER CHECK (l0_flagged IN (0,1)),
  l0_rule_hits        TEXT,              -- JSON array of rule names that fired

  -- ==========================================================================
  -- LAYER 1 — heuristics
  -- Each signal gets its own column: they are queryable, and in Week 4 they
  -- become classifier features directly. l1_features_json is the escape hatch
  -- for signals you invent later, so adding one never means a migration.
  -- ==========================================================================
  l1_verdict                 TEXT CHECK (l1_verdict IN ('pass','fail','uncertain')),
  l1_token_overlap           REAL,       -- answer ↔ context, 0..1
  l1_max_similarity          REAL,       -- best retrieval score
  l1_mean_similarity         REAL,
  l1_score_spread            REAL,       -- max minus min across the k scores
  l1_min_sentence_similarity REAL,       -- weakest sentence in the answer — the workhorse signal
  l1_answer_len_tokens       INTEGER,
  l1_context_len_tokens      INTEGER,
  l1_len_ratio               REAL,       -- answer ÷ context
  l1_hedging_count           INTEGER,    -- "it appears", "generally", "may be"…
  l1_unsupported_numerics    INTEGER,    -- numbers/dates in the answer, absent from context
  l1_pii_hits                INTEGER,
  l1_toxicity_hits           INTEGER,
  l1_features_json           TEXT,       -- any additional signals, as JSON

  -- ==========================================================================
  -- LAYER 2 — trained classifier
  -- Store raw AND calibrated probability: the gap between them is the
  -- calibration story, and the cascade routes on the calibrated one.
  -- l2_features_json is the exact vector fed to the model — without it a
  -- prediction is not reproducible.
  -- ==========================================================================
  l2_verdict                 TEXT CHECK (l2_verdict IN ('pass','fail','uncertain')),
  l2_probability_raw         REAL,
  l2_probability_calibrated  REAL,
  l2_model_version           TEXT,
  l2_features_json           TEXT,

  -- ==========================================================================
  -- LAYER 3 — LLM judge
  -- l3_prompt_version matters more than it looks: change the judge prompt and
  -- older rows are no longer comparable. Keep the raw response so a parser bug
  -- never costs you a re-run.
  -- ==========================================================================
  l3_verdict           TEXT CHECK (l3_verdict IN ('pass','fail','uncertain')),
  l3_groundedness      REAL,
  l3_relevance         REAL,
  l3_unsupported_span  TEXT,             -- the specific claim the judge couldn't support
  l3_rationale         TEXT,
  l3_judge_model       TEXT,
  l3_prompt_version    TEXT,
  l3_raw_response      TEXT,

  -- ==========================================================================
  -- OUTCOME
  -- human_label is ground truth: 1 = hallucination, 0 = grounded.
  -- terminating_layer is where the cascade WOULD have stopped — recorded even
  -- in shadow mode, where every layer ran anyway. Cross-tabbing these two is
  -- the miss-attribution analysis.
  -- ==========================================================================
  human_label         INTEGER CHECK (human_label IN (0,1)),
  human_label_notes   TEXT,
  human_labeled_at    TEXT,
  judge_agreed        INTEGER CHECK (judge_agreed IN (0,1)),  -- l3_verdict vs human_label
  terminating_layer   TEXT CHECK (terminating_layer IN ('L0','L1','L2','L3')),
  cascade_verdict     TEXT CHECK (cascade_verdict IN ('pass','fail')),

  -- ==========================================================================
  -- OPERATIONS
  -- Latency and cost are split per stage because the whole argument is about
  -- WHERE the time and money go. Embedding and classifier costs are included
  -- deliberately: reporting only the judge's tokens inflates the saving, and
  -- that is exactly the shortcut an interviewer probes for.
  -- ==========================================================================
  latency_retrieval_ms  REAL,
  latency_generation_ms REAL,
  latency_l0_ms         REAL,
  latency_l1_ms         REAL,
  latency_l2_ms         REAL,
  latency_l3_ms         REAL,

  cost_embedding_usd    REAL,
  cost_generation_usd   REAL,
  cost_l1_usd           REAL,
  cost_l2_usd           REAL,
  cost_l3_usd           REAL,

  error                 TEXT   -- non-null if the row failed partway; keep the row, don't drop it
);

-- ============================================================================
-- INDEXES — the columns you will actually GROUP BY and JOIN on.
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_qlog_run         ON query_log (run_id);
CREATE INDEX IF NOT EXISTS idx_qlog_question    ON query_log (question_id);
CREATE INDEX IF NOT EXISTS idx_qlog_config      ON query_log (pipeline_config_id);
CREATE INDEX IF NOT EXISTS idx_qlog_mode        ON query_log (induction_mode);
CREATE INDEX IF NOT EXISTS idx_qlog_terminating ON query_log (terminating_layer);
CREATE INDEX IF NOT EXISTS idx_qlog_label       ON query_log (human_label);

-- One question should appear at most once per run.
CREATE UNIQUE INDEX IF NOT EXISTS idx_qlog_run_question ON query_log (run_id, question_id);


-- ============================================================================
-- SANITY CHECKS — not the analysis layer, just proof the schema can answer the
-- questions it was designed for. Run these against a handful of rows in Week 1.
-- ============================================================================

-- Escalation rate (the headline number):
--   SELECT terminating_layer, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS share
--   FROM query_log WHERE run_id = ? GROUP BY terminating_layer;

-- Cost per query, cascade vs judge-everything (fixed in T9, see src/sanity_queries.py):
-- the cascade pays only for the layers up to terminating_layer. Summing ALL layers counts
-- judge calls that shadow mode made but the cascade never would; the first version of this
-- query made the cascade look MORE expensive than judging everything.
--   SELECT AVG(COALESCE(cost_embedding_usd,0) + CASE terminating_layer
--            WHEN 'L1' THEN COALESCE(cost_l1_usd,0)
--            WHEN 'L2' THEN COALESCE(cost_l1_usd,0) + COALESCE(cost_l2_usd,0)
--            WHEN 'L3' THEN COALESCE(cost_l1_usd,0) + COALESCE(cost_l2_usd,0) + COALESCE(cost_l3_usd,0)
--          END) AS cascade_cost,
--          AVG(COALESCE(cost_embedding_usd,0) + COALESCE(cost_l3_usd,0)) AS judge_every_cost
--   FROM query_log WHERE run_id = ? AND error IS NULL AND shadow_mode = 1;

-- Miss attribution — of the hallucinations we let through, who cleared them:
--   SELECT terminating_layer, COUNT(*) FROM query_log
--   WHERE human_label = 1 AND cascade_verdict = 'pass'
--   GROUP BY terminating_layer;

-- Detection rate per induction mode (where the honest finding lives):
--   SELECT induction_mode,
--          SUM(CASE WHEN cascade_verdict = 'fail' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS caught
--   FROM query_log WHERE human_label = 1 GROUP BY induction_mode;
