import psycopg2

def connect_db():
    return psycopg2.connect(
        host="localhost",
        database="proctor",
        user="postgres",
        password="1234"
    )

def insert_log(time, status, warnings):
    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO logs (time, status, warnings) VALUES (%s,%s,%s)",
        (time, status, warnings)
    )

    conn.commit()
    cur.close()
    conn.close()