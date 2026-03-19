CREATE TABLE IF NOT EXISTS mailkb.emails_unique
(
    id String,
    thread_key String,
    message_id String,
    subject String,
    from_addr Array(String),
    to_addr Array(String),
    cc_addr Array(String),
    bcc_addr Array(String),
    sent_at_utc DateTime,
    folder String,
    body_text String
)
ENGINE = MergeTree
ORDER BY (thread_key, sent_at_utc);
