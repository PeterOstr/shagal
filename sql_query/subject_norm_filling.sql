ALTER TABLE mailkb.emails
UPDATE subject_norm =
    if(
        trim(
            replaceRegexpAll(
                lower(subject),
                '^((re|fw|fwd|ответ|aw|wg):\\s*)+',
                ''
            )
        ) = '',
        'no_subject',
        trim(
            replaceRegexpAll(
                lower(subject),
                '^((re|fw|fwd|ответ|aw|wg):\\s*)+',
                ''
            )
        )
    )
WHERE 1;