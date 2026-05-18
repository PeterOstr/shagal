CREATE TABLE mailkb.emails
(
    id String,
    message_id String,
    thread_id String,
    subject String,
    from_addr Array(String),
    to_addr Array(String),
    cc_addr Array(String),
    bcc_addr Array(String),
    sent_at_utc DateTime,
    sent_at_raw String,
    folder String,
    body_text String,
    body_html String
)
ENGINE = MergeTree
ORDER BY id;