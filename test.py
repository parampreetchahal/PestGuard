import mysql.connector

def test_mysql_connection():
    try:
        print("Connecting to MySQL...")

        conn = mysql.connector.connect(
            host="sql12.freesqldatabase.com",        # e.g., sql11.freesqldatabase.com
            user="sql12811476",    # e.g., sql1234567
            password="WKkgyF1mBM",
            database="sql12811476",
            port=3306
        )

        print("Connected successfully!")

        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE();")
        db_name = cursor.fetchone()

        print("Active Database:", db_name[0])

        cursor.close()
        conn.close()
        print("Connection closed.")

    except Exception as e:
        print("ERROR:", e)

# run test
test_mysql_connection()
