import psycopg2
from psycopg2 import sql
from contextlib import closing

class DatabaseController:
    def __init__(self, dbname="meta_learning", user="user", password="1234", host="localhost", port="5432"):
        self.db_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }

    def execute_query(self, query, params=None, fetch=False):
        with closing(psycopg2.connect(**self.db_params)) as conn, closing(conn.cursor()) as cursor:
            try:
                cursor.execute(query, params)
                if fetch:
                    return cursor.fetchall()
                conn.commit()
            except Exception as e:
                conn.rollback()
                print("error:", e)

    def create_table(self, table_name, columns):
        columns_str = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        query = sql.SQL(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});")
        self.execute_query(query)

    def insert_data(self, table_name, data):
        columns = ", ".join(data.keys())
        values_placeholders = ", ".join(["%s"] * len(data))
        query = sql.SQL(f"INSERT INTO {table_name} ({columns}) VALUES ({values_placeholders});")
        self.execute_query(query, tuple(data.values()))

    def fetch_data(self, table_name, conditions=None):
        base_query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
        
        if conditions:
            where_clause = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(
                sql.Composed([sql.Identifier(col), sql.SQL("= %s")]) for col in conditions.keys()
            )
            query = base_query + where_clause
            return self.execute_query(query, tuple(conditions.values()), fetch=True)
        
        return self.execute_query(base_query, fetch=True)

    def clear_table(self, table_name):
        query = sql.SQL(f"DELETE FROM {table_name};")
        self.execute_query(query)

    def get_all_data(self, table_name):
        query = sql.SQL(f"SELECT * FROM {table_name};")
        return self.execute_query(query, fetch=True)

    def get_data_length(self, table_name):
        query = sql.SQL(f"SELECT COUNT(*) FROM {table_name};")
        result = self.execute_query(query, fetch=True)
        return result[0][0] if result else 0
    
    def table_exists(self, table_name):
        query = sql.SQL("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);")
        result = self.execute_query(query, (table_name,), fetch=True)
        return result[0][0] if result else False
    
    def get_columns(self, table_name):
        query = sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s;")
        result = self.execute_query(query, (table_name,), fetch=True)
        return [row[0] for row in result] if result else []
    
    def delete_by_condition(self, table_name, conditions):    
        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        query = sql.SQL(f"DELETE FROM {table_name} WHERE {where_clause};")
        self.execute_query(query, tuple(conditions.values()))

    def update_data(self, table_name, updates, conditions):
        set_clause = ", ".join([f"{col} = %s" for col in updates.keys()])
        where_clause = " AND ".join([f"{col} = %s" for col in conditions.keys()])
        query = sql.SQL(f"UPDATE {table_name} SET {set_clause} WHERE {where_clause};")
        self.execute_query(query, tuple(updates.values()) + tuple(conditions.values()))
        
    def execute_raw_query(self, query, params=None):
        return self.execute_query(query, params, fetch=True)

