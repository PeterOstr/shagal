CREATE TABLE mailkb.email_clean
(
    email_id String,
    thread_id String,
    body_clean String,
    model_name String,
    parser_version String,
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree
ORDER BY (email_id);