ALTER TABLE mailkb.emails
ADD COLUMN thread_key String;

ALTER TABLE mailkb.emails
UPDATE thread_key =
lower(
    replaceRegexpAll(
        subject,
        '^(?i)((re|fw|fwd|ответ|aw)\\s*:\\s*)+',
        ''
    )
)
WHERE 1