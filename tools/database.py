"""
An example tool for connecting to and querying a database.

This module provides a template for a tool that can execute SQL queries
against a database (e.g., SQLite, PostgreSQL). This allows the agent to
answer questions by querying structured data.
"""
import sqlite3
from typing import List, Dict, Any
from ..utils.logging import get_logger

logger = get_logger(__name__)

class DatabaseConnector:
    """
    A tool for executing read-only SQL queries.
    """
    def __init__(self, connection_string: str):
        """
        Initializes the DatabaseConnector.

        Args:
            connection_string (str): The database connection string.
                                     For SQLite, this is the path to the .db file.
        """
        # TODO: Add support for other database backends like PostgreSQL or MySQL.
        # This would require using libraries like psycopg2 or mysql-connector-python.
        self.connection_string = connection_string
        self.conn = None
        try:
            if self.connection_string.startswith("sqlite:///"):
                db_file = self.connection_string.replace("sqlite:///", "")
                self.conn = sqlite3.connect(db_file, check_same_thread=False)
                logger.info(f"Database connector initialized for SQLite DB: {db_file}")
            else:
                logger.warning(f"Unsupported database connection string: {connection_string}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")

    def execute(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a read-only SQL query.

        For safety, this should only execute SELECT statements.

        Args:
            query (str): The SQL query to execute.

        Returns:
            List[Dict[str, Any]]: A list of rows, where each row is a dictionary.
                                  Returns an error message if the query fails.
        """
        if not self.conn:
            logger.error("Cannot execute query: database connection is not available.")
            return [{"error": "Database connection not established."}]

        # TODO: Implement a more robust safety check to prevent destructive queries (DELETE, UPDATE, etc.).
        # A simple check like this is not foolproof.
        if not query.strip().upper().startswith("SELECT"):
            logger.warning(f"Blocked non-SELECT query: {query}")
            return [{"error": "Only SELECT queries are allowed."}]

        logger.debug(f"Executing SQL query: '{query}'")
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            logger.info(f"SQL query returned {len(results)} rows.")
            return results
        except Exception as e:
            logger.error(f"Error executing SQL query '{query}': {e}")
            return [{"error": f"Query failed: {e}"}]
        finally:
            # This simple connector doesn't close the connection to allow multiple queries.
            # In a real app, connection pooling should be used.
            pass

    def close(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")

# Example usage:
# if __name__ == '__main__':
#     # Create a dummy database for testing
#     conn = sqlite3.connect("test.db")
#     conn.execute("CREATE TABLE IF NOT EXISTS users (id INT, name TEXT, city TEXT)")
#     conn.execute("INSERT INTO users VALUES (1, 'Alice', 'New York'), (2, 'Bob', 'London')")
#     conn.commit()
#     conn.close()

#     db_tool = DatabaseConnector("sqlite:///test.db")
#     print(db_tool.execute("SELECT name, city FROM users WHERE city = 'New York'"))
#     print(db_tool.execute("DELETE FROM users")) # Should be blocked
#     db_tool.close()
