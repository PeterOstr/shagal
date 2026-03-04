ALTER TABLE mailkb.emails
ADD COLUMN thread_id String;

ALTER TABLE mailkb.emails
UPDATE thread_id =
    if(
        subject_norm = 'no_subject',
        md5(
            concat(
                'nosubj_',
                toString(toDate(sent_at_utc))
            )
        ),
        md5(subject_norm)
    )
WHERE 1;