INSERT INTO mailkb.threads
SELECT
    thread_id,
    subject_norm,
    count() AS emails_count,
    min(sent_at_utc) AS first_sent_at,
    max(sent_at_utc) AS last_sent_at
FROM mailkb.emails
GROUP BY
    thread_id,
    subject_norm;