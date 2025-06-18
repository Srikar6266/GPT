import aiosqlite
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "documents.db"

async def init_db():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute('''CREATE TABLE IF NOT EXISTS documents
                                (id INTEGER PRIMARY KEY, filename TEXT, text TEXT, summary TEXT)''')
            await db.commit()
            logger.info("DB initialized, ready to rock!")
    except Exception as e:
        logger.error(f"DB init failed: {e}")
        raise

async def store_document(filename, text, summary):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT INTO documents (filename, text, summary) VALUES (?, ?, ?)",
                             (filename, text, summary))
            await db.commit()
            logger.info(f"Stored doc: {filename}")
    except Exception as e:
        logger.error(f"Store doc failed: {e}")
        raise

async def get_document_text(filename):
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT text FROM documents WHERE filename = ?", (filename,)) as cursor:
                result = await cursor.fetchone()
                if result:
                    logger.info(f"Retrieved text for: {filename}")
                    return result[0]
                logger.warning(f"No doc found: {filename}")
                return None
    except Exception as e:
        logger.error(f"Get doc failed: {e}")
        return None