CREATE TABLE mailkb.attachments
(
    email_id String,
    filename String,
    path String,
    size_bytes UInt64
)
ENGINE = MergeTree
ORDER BY email_id;