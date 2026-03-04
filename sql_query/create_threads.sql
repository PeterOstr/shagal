CREATE TABLE mailkb.threads
(
    thread_id String,
    subject_norm String,
    emails_count UInt32,
    first_sent_at DateTime64(3, 'UTC'),
    last_sent_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree
ORDER BY thread_id;