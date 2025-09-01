# pipeline/database_connector.py - Database Connection Management
import pandas as pd
import logging
from typing import Dict, List, Optional
from sqlalchemy import create_engine, text
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """Handles database connections and data loading with proper resource management"""
    
    def __init__(self):
        self.engine = None
        self.connected = False
    
    def connect(self, connection_string: str, chunk_size: int = 10000) -> bool:
        """Connect to database with proper resource management"""
        try:
            # Parse connection string to hide sensitive info in logs
            parsed_url = urlparse(connection_string)
            safe_connection_info = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}/{parsed_url.path.lstrip('/')}"
            logger.info(f"Connecting to database: {safe_connection_info}")

            # Create database connector with connection pooling
            self.engine = create_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Test connection
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            self.connected = True
            logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            self.connected = False
            return False
    
    def get_table_list(self) -> List[str]:
        """Get list of available tables"""
        if not self.connected:
            raise RuntimeError("Not connected to database")
        
        try:
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            
            # Limit number of tables to prevent memory issues
            if len(tables) > 50:
                logger.warning(f"Large number of tables ({len(tables)}). Consider specifying specific tables.")
                tables = tables[:50]
            
            return tables
            
        except Exception as e:
            logger.error(f"Error getting table list: {e}")
            return []
    
    def load_table_data(self, table_name: str, chunk_size: int = 10000) -> pd.DataFrame:
        """Load data from a specific table with chunking for large datasets"""
        if not self.connected:
            raise RuntimeError("Not connected to database")
        
        try:
            with self.engine.connect() as connection:
                # First, check table size
                count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                row_count = connection.execute(count_query).scalar()
                
                logger.info(f"Loading table {table_name} ({row_count} rows)")
                
                if row_count > 100000:
                    logger.warning(f"Table {table_name} has {row_count} rows. Loading in chunks.")
                    # Load in chunks for large tables
                    chunks = []
                    for chunk in pd.read_sql_table(table_name, connection, chunksize=chunk_size):
                        chunks.append(chunk)
                        if len(chunks) * chunk_size > 1000000:  # Limit to 1M rows max
                            logger.warning(f"Table {table_name} truncated to 1M rows for memory safety")
                            break
                    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                else:
                    # Load entire table for smaller datasets
                    return pd.read_sql_table(table_name, connection)
                    
        except Exception as e:
            logger.error(f"Error loading table {table_name}: {e}")
            return pd.DataFrame()
    
    def load_tables(self, table_names: Optional[List[str]] = None, chunk_size: int = 10000) -> Dict[str, pd.DataFrame]:
        """Load multiple tables"""
        if not self.connected:
            raise RuntimeError("Not connected to database")
        
        if table_names is None:
            table_names = self.get_table_list()
        
        data = {}
        for table_name in table_names:
            try:
                data[table_name] = self.load_table_data(table_name, chunk_size)
                logger.info(f"Loaded table {table_name}: {len(data[table_name])} rows, {len(data[table_name].columns)} columns")
            except Exception as e:
                logger.error(f"Failed to load table {table_name}: {e}")
                continue
        
        return data
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self.connected = False
            logger.info("Database connection closed")