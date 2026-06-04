CREATE TABLE IF NOT EXISTS mailkb.mail_parsed
(
    email_id String,
    parsed_json String
)
ENGINE = MergeTree
ORDER BY email_id;