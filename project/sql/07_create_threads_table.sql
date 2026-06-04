DROP TABLE IF EXISTS mailkb.threads;

CREATE TABLE mailkb.threads
(
    thread_key String,
    subject_norm String,
    subject String,
    unique_emails_count UInt32,
    first_sent_at DateTime64(3, 'UTC'),
    last_sent_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree
ORDER BY thread_key;