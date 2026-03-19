INSERT INTO mailkb.threads
SELECT
    thread_key,
    lowerUTF8(
        trim(
            replaceRegexpAll(
                replaceRegexpAll(
                    latest_subject,
                    '^(?:(?:[Rr][Ee]|[Ff][Ww][Dd]?|[Aa][Ww]|[Оо]твет):\\s*)+',
                    ''
                ),
                '\\s+',
                ' '
            )
        )
    ) AS subject_norm,
    latest_subject AS subject,
    unique_emails_count,
    first_sent_at,
    last_sent_at
FROM
(
    SELECT
        thread_key,
        argMax(subject, sent_at_utc) AS latest_subject,
        toUInt32(count()) AS unique_emails_count,
        min(sent_at_utc) AS first_sent_at,
        max(sent_at_utc) AS last_sent_at
    FROM mailkb.emails_unique
    WHERE thread_key != ''
    GROUP BY thread_key
);