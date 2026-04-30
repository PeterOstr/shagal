import clickhouse_connect
from app.config import settings

class ClickhouseRepository:

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=settings.CLICKHOUSE_HOST,
                port=settings.CLICKHOUSE_PORT,
                username=settings.CLICKHOUSE_USER,
                password=settings.CLICKHOUSE_PASSWORD,
            )
        return self._client

    def fetch_emails(self, limit=100, offset=0):
        query = f"""
            SELECT id, message_id, subject, from_addr, to_addr, cc_addr, bcc_addr,
                   sent_at_utc, folder, body_text, body_html
            FROM mailkb.emails
            ORDER BY sent_at_utc DESC
            LIMIT {limit} OFFSET {offset}
        """
        return self.client.query_df(query)
