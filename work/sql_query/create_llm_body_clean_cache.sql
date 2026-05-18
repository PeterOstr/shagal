CREATE TABLE mailkb.llm_body_clean_cache
(
    raw_md5 String,
    parser_version String,
    model_name String,
    status String,
    body_clean String,
    error String,
    tokens_in UInt32,
    tokens_out UInt32,
    latency_ms UInt32,
    created_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree
ORDER BY (raw_md5, parser_version);