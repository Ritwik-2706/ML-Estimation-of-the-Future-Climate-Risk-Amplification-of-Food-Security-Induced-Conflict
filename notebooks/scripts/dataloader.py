import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set

@dataclass
class DataLoader:
    """
    DataLoader is a class to interact with SQLite database.
    It provides several methods to get information from database
    like table names, column names, and commodity related data.
    
    Usage:
    1. Instantiate the DataLoader with your SQLite db_path.
    2. Use the DataLoader methods to interact with your database.
    """
    db_path: str

    def __post_init__(self):
        """
        This method is automatically invoked after the class is instantiated.
        It connects to the SQLite database and initialises the cursor and tables.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.tables = self.get_tables()

    def get_tables(self) -> List[str]:
        """
        Returns a list of table names in the database.
        """
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [table[0] for table in self.cursor.fetchall()]

    def get_table_columns(self, table: str) -> List[str]:
        """
        Given a table name, returns a list of column names in the table.
        """
        self.cursor.execute(f"PRAGMA table_info({table});")
        return [column[1] for column in self.cursor.fetchall()]

    def get_commodity_names(self) -> Set[str]:
        """
        Returns a set of commodity names by parsing the column names 
        in the 'cmo_historical_data_monthly' table.
        """
        columns = self.get_table_columns("cmo_historical_data_monthly")
        columns = [column.split(',')[0].split()[0].lower() for column in columns if column != "MONTH"]
        return set(columns)

    def get_commodity_historical_monthly_data(self, commodity: str) -> pd.DataFrame:
        """
        Given a commodity name, returns a DataFrame containing historical 
        monthly data for that commodity.
        """
        commodity_columns = self.get_table_columns("cmo_historical_data_monthly")
        commodity_columns = [comm for comm in commodity_columns if commodity in comm.lower()]
        commodity_columns_quoted = [f"`{col}`" for col in commodity_columns]  # Adding backticks around column names

        query = f"SELECT MONTH, {', '.join(commodity_columns_quoted)} FROM cmo_historical_data_monthly"

        df = pd.read_sql(query, self.conn)
        return df

    @staticmethod
    def close_connection():
        """
        Closes the connection to the SQLite database.
        """
        self.conn.close()