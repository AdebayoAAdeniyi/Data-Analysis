"""
Design, build and query a relational database with MySQL from Python.

The scenario is a language school. Teachers teach courses, courses are bought
by client companies, and participants (employees of the clients) take
courses. The script shows the full life cycle of a small database:

    1. Connect safely (credentials from environment variables, never in code)
    2. Create the database and five related tables with primary keys,
       foreign keys, UNIQUE and NOT NULL constraints
    3. Populate the tables (single INSERT and executemany with placeholders)
    4. Read data into pandas DataFrames, with column names taken from the
       cursor instead of being typed by hand
    5. JOINs (INNER, LEFT, RIGHT) and a demonstration of when they differ
    6. Aggregation with GROUP BY / HAVING for simple business questions
    7. Parameterised UPDATE (protection against SQL injection)
    8. DELETE with ON DELETE CASCADE and ON DELETE SET NULL in action
    9. Why DROP TABLE fails when another table references it
   10. A transaction that is rolled back when one statement fails

Requirements: a running MySQL 8 (or MariaDB 10.5+) server and
``pip install mysql-connector-python pandas``.

Usage::

    export MYSQL_HOST=localhost MYSQL_USER=your_user MYSQL_PASSWORD=your_password
    python school_database.py --reset

``--reset`` drops and rebuilds the demo database (default name
``school_demo``) so the script can be run any number of times.
"""
from __future__ import annotations

import argparse
import os
from contextlib import closing

import mysql.connector
import pandas as pd
from mysql.connector import Error

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


# --------------------------------------------------------------------------- #
# Connection and helpers
# --------------------------------------------------------------------------- #
class Database:
    """Thin wrapper around a mysql.connector connection."""

    def __init__(self, host: str, user: str, password: str, database: str | None = None):
        self.conn = mysql.connector.connect(host=host, user=user, password=password,
                                            database=database)
        info = getattr(self.conn, "server_info", None) or self.conn.get_server_info()
        print(f"Connected to MySQL server {info} on {host}"
              + (f", database '{database}'" if database else ""))

    def execute(self, sql: str, params: tuple | None = None, quiet: bool = False) -> int:
        """Run one statement and commit. Returns the number of affected rows."""
        with closing(self.conn.cursor()) as cur:
            cur.execute(sql, params)
            self.conn.commit()
            if not quiet:
                print(f"  OK ({cur.rowcount} row(s) affected): {first_line(sql)}")
            return cur.rowcount

    def executemany(self, sql: str, rows: list[tuple]) -> int:
        with closing(self.conn.cursor()) as cur:
            cur.executemany(sql, rows)
            self.conn.commit()
            print(f"  OK ({cur.rowcount} row(s) inserted): {first_line(sql)}")
            return cur.rowcount

    def query(self, sql: str, params: tuple | None = None) -> pd.DataFrame:
        """Run a SELECT and return a DataFrame. Column names come from
        cursor.description, so they always match the query."""
        with closing(self.conn.cursor()) as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols)

    def close(self) -> None:
        self.conn.close()


def first_line(sql: str) -> str:
    return " ".join(sql.split())[:70]


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n--- {title} ---")
    print(df.to_string(index=False))


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS teacher (
        teacher_id   INT PRIMARY KEY,
        first_name   VARCHAR(40) NOT NULL,
        last_name    VARCHAR(40) NOT NULL,
        language_1   VARCHAR(3)  NOT NULL,
        language_2   VARCHAR(3),
        dob          DATE,
        tax_id       INT UNIQUE,
        phone_no     VARCHAR(20)
    )""",
    """
    CREATE TABLE IF NOT EXISTS client (
        client_id    INT PRIMARY KEY,
        client_name  VARCHAR(40) NOT NULL,
        address      VARCHAR(60) NOT NULL,
        industry     VARCHAR(20)
    )""",
    """
    CREATE TABLE IF NOT EXISTS participant (
        participant_id INT PRIMARY KEY,
        first_name     VARCHAR(40) NOT NULL,
        last_name      VARCHAR(40) NOT NULL,
        phone_no       VARCHAR(20),
        client         INT,
        CONSTRAINT fk_participant_client FOREIGN KEY (client)
            REFERENCES client(client_id) ON DELETE SET NULL
    )""",
    """
    CREATE TABLE IF NOT EXISTS course (
        course_id           INT PRIMARY KEY,
        course_name         VARCHAR(40) NOT NULL,
        language            VARCHAR(3)  NOT NULL,
        level               VARCHAR(2),
        course_length_weeks INT,
        start_date          DATE,
        in_school           BOOLEAN,
        teacher             INT,
        client              INT,
        CONSTRAINT fk_course_teacher FOREIGN KEY (teacher)
            REFERENCES teacher(teacher_id) ON DELETE SET NULL,
        CONSTRAINT fk_course_client FOREIGN KEY (client)
            REFERENCES client(client_id) ON DELETE SET NULL
    )""",
    # Junction table for the many to many relationship participant <-> course
    """
    CREATE TABLE IF NOT EXISTS takes_course (
        participant_id INT,
        course_id      INT,
        PRIMARY KEY (participant_id, course_id),
        CONSTRAINT fk_takes_participant FOREIGN KEY (participant_id)
            REFERENCES participant(participant_id) ON DELETE CASCADE,
        CONSTRAINT fk_takes_course FOREIGN KEY (course_id)
            REFERENCES course(course_id) ON DELETE CASCADE
    )""",
]

# --------------------------------------------------------------------------- #
# Data (fictional, from the freeCodeCamp tutorial cited in the README)
# --------------------------------------------------------------------------- #
TEACHERS = [
    (1, "James", "Smith", "ENG", None, "1985-04-20", 12345, "+491774553676"),
    (2, "Stefanie", "Martin", "FRA", None, "1970-02-17", 23456, "+491234567890"),
    (3, "Steve", "Wang", "MAN", "ENG", "1990-11-12", 34567, "+447840921333"),
    (4, "Friederike", "Müller-Rossi", "DEU", "ITA", "1987-07-07", 45678, "+492345678901"),
    (5, "Isobel", "Ivanova", "RUS", "ENG", "1963-05-30", 56789, "+491772635467"),
    (6, "Niamh", "Murphy", "ENG", "IRI", "1995-09-08", 67890, "+491231231232"),
]
CLIENTS = [
    (101, "Big Business Federation", "123 Falschungstraße, 10999 Berlin", "NGO"),
    (102, "eCommerce GmbH", "27 Ersatz Allee, 10317 Berlin", "Retail"),
    (103, "AutoMaker AG", "20 Künstlichstraße, 10023 Berlin", "Auto"),
    (104, "Banko Bank", "12 Betrugstraße, 12345 Berlin", "Banking"),
    (105, "WeMoveIt GmbH", "138 Arglistweg, 10065 Berlin", "Logistics"),
]
PARTICIPANTS = [
    (101, "Marina", "Berg", "491635558182", 101), (102, "Andrea", "Duerr", "49159555740", 101),
    (103, "Philipp", "Probst", "49155555692", 102), (104, "René", "Brandt", "4916355546", 102),
    (105, "Susanne", "Shuster", "49155555779", 102), (106, "Christian", "Schreiner", "49162555375", 101),
    (107, "Harry", "Kim", "49177555633", 101), (108, "Jan", "Nowak", "49151555824", 101),
    (109, "Pablo", "Garcia", "49162555176", 101), (110, "Melanie", "Dreschler", "49151555527", 103),
    (111, "Dieter", "Durr", "49178555311", 103), (112, "Max", "Mustermann", "49152555195", 104),
    (113, "Maxine", "Mustermann", "49177555355", 104), (114, "Heiko", "Fleischer", "49155555581", 105),
]
COURSES = [
    (12, "English for Logistics", "ENG", "A1", 10, "2020-02-01", True, 1, 105),
    (13, "Beginner English", "ENG", "A2", 40, "2019-11-12", False, 6, 101),
    (14, "Intermediate English", "ENG", "B2", 40, "2019-11-12", False, 6, 101),
    (15, "Advanced English", "ENG", "C1", 40, "2019-11-12", False, 6, 101),
    (16, "Mandarin für Autoindustrie", "MAN", "B1", 15, "2020-01-15", True, 3, 103),
    (17, "Français intermédiaire", "FRA", "B1", 18, "2020-04-03", False, 2, 101),
    (18, "Deutsch für Anfänger", "DEU", "A2", 8, "2020-02-14", True, 4, 102),
    (19, "Intermediate English", "ENG", "B2", 10, "2020-03-29", False, 1, 104),
    (20, "Fortgeschrittenes Russisch", "RUS", "C1", 4, "2020-04-08", False, 5, 103),
]
TAKES_COURSE = [(101, 15), (101, 17), (102, 17), (103, 18), (104, 18), (105, 18),
                (106, 13), (107, 13), (108, 13), (109, 14), (109, 15), (110, 16),
                (110, 20), (111, 16), (114, 12), (112, 19), (113, 19)]


def placeholders(n: int) -> str:
    return ", ".join(["%s"] * n)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def build(db: Database) -> None:
    print("\n[2] Creating tables")
    for ddl in SCHEMA:
        db.execute(ddl)

    print("\n[3] Populating tables (parameterised executemany)")
    for table, rows in [("teacher", TEACHERS), ("client", CLIENTS),
                        ("participant", PARTICIPANTS), ("course", COURSES),
                        ("takes_course", TAKES_COURSE)]:
        # INSERT IGNORE skips rows whose primary key already exists, so a
        # second run without --reset does not crash.
        db.executemany(f"INSERT IGNORE INTO {table} VALUES ({placeholders(len(rows[0]))})",
                       rows)
    show("Tables in the database", db.query("SHOW TABLES"))


def read_and_join(db: Database) -> None:
    print("\n[4] Reading data into pandas")
    show("All teachers", db.query("SELECT * FROM teacher"))

    print("\n[5] JOINs")
    show("Courses taught off site, with the client's address", db.query("""
        SELECT c.course_id, c.course_name, c.language, cl.client_name, cl.address
        FROM course AS c
        JOIN client AS cl ON c.client = cl.client_id
        WHERE c.in_school = FALSE
        ORDER BY c.course_id"""))

    show("Participants with the courses they take and the teacher (4 table join)", db.query("""
        SELECT p.first_name, p.last_name, cl.client_name, c.course_name,
               CONCAT(t.first_name, ' ', t.last_name) AS teacher
        FROM participant AS p
        JOIN takes_course AS tc ON tc.participant_id = p.participant_id
        JOIN course       AS c  ON c.course_id = tc.course_id
        LEFT JOIN teacher AS t  ON t.teacher_id = c.teacher
        LEFT JOIN client  AS cl ON cl.client_id = p.client
        ORDER BY p.participant_id"""))

    # With the original data every client has a course and every course has a
    # client, so INNER, LEFT and RIGHT joins return the same 9 rows. Adding a
    # client without courses makes the difference visible.
    db.execute("INSERT IGNORE INTO client VALUES (106, 'Startup Labs', "
               "'5 Probeweg, 10115 Berlin', 'Tech')")
    for kind in ["INNER", "LEFT", "RIGHT"]:
        df = db.query(f"""
            SELECT cl.client_id, cl.client_name, c.course_name
            FROM client AS cl
            {kind} JOIN course AS c ON cl.client_id = c.client
            ORDER BY cl.client_id, c.course_id""")
        show(f"{kind} JOIN client -> course ({len(df)} rows)", df)
    print("\nLEFT JOIN keeps 'Startup Labs' with a NULL course; INNER JOIN drops it.")


def aggregate(db: Database) -> None:
    print("\n[6] Aggregation: simple business questions")
    show("How many participants does each course have?", db.query("""
        SELECT c.course_id, c.course_name, c.level, COUNT(tc.participant_id) AS participants
        FROM course AS c
        LEFT JOIN takes_course AS tc ON tc.course_id = c.course_id
        GROUP BY c.course_id, c.course_name, c.level
        ORDER BY participants DESC, c.course_id"""))

    show("Teacher workload: number of courses and total teaching weeks", db.query("""
        SELECT CONCAT(t.first_name, ' ', t.last_name) AS teacher,
               COUNT(c.course_id) AS courses,
               COALESCE(SUM(c.course_length_weeks), 0) AS total_weeks
        FROM teacher AS t
        LEFT JOIN course AS c ON c.teacher = t.teacher_id
        GROUP BY t.teacher_id, t.first_name, t.last_name
        ORDER BY total_weeks DESC"""))

    show("Clients buying more than one course (GROUP BY ... HAVING)", db.query("""
        SELECT cl.client_name, cl.industry, COUNT(*) AS courses_bought,
               GROUP_CONCAT(DISTINCT c.language ORDER BY c.language) AS languages
        FROM client AS cl
        JOIN course AS c ON c.client = cl.client_id
        GROUP BY cl.client_id, cl.client_name, cl.industry
        HAVING COUNT(*) > 1
        ORDER BY courses_bought DESC"""))

    show("Teachers older than the average teacher (subquery)", db.query("""
        SELECT first_name, last_name, dob,
               TIMESTAMPDIFF(YEAR, dob, '2020-01-01') AS age_on_2020_01_01
        FROM teacher
        WHERE dob < (SELECT FROM_DAYS(AVG(TO_DAYS(dob))) FROM teacher)
        ORDER BY dob"""))


def modify(db: Database) -> None:
    print("\n[7] Parameterised UPDATE (values are passed separately from the SQL text,")
    print("    so user input can never change the statement: no SQL injection)")
    db.execute("UPDATE client SET address = %s WHERE client_id = %s",
               ("23 Fingiertweg, 14534 Berlin", 101))
    show("Client 101 after the update",
         db.query("SELECT * FROM client WHERE client_id = %s", (101,)))

    print("\nAdding two teachers with executemany")
    db.executemany(
        "INSERT IGNORE INTO teacher (teacher_id, first_name, last_name, language_1, "
        "language_2, dob, tax_id, phone_no) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        [(7, "Hank", "Dodson", "ENG", None, "1991-12-23", 11111, "+491772345678"),
         (8, "Sue", "Perkins", "MAN", "ENG", "1976-02-02", 22222, "+491443456432")])

    print("\n[8] DELETE and referential actions")
    before = db.query("SELECT COUNT(*) AS n FROM takes_course WHERE course_id = 20")["n"][0]
    db.execute("DELETE FROM course WHERE course_id = %s", (20,))
    after = db.query("SELECT COUNT(*) AS n FROM takes_course WHERE course_id = 20")["n"][0]
    print(f"  Enrolments in course 20: {before} before, {after} after "
          "(removed automatically by ON DELETE CASCADE)")

    db.execute("DELETE FROM teacher WHERE teacher_id = %s", (1,))
    show("Teacher 1 (James Smith) deleted: his courses survive with teacher = NULL "
         "(ON DELETE SET NULL)",
         db.query("SELECT course_id, course_name, teacher FROM course "
                  "WHERE course_id IN (12, 19)"))
    n = db.execute("DELETE FROM course WHERE course_id = %s", (2034,))
    print(f"  Deleting a course id that does not exist affects {n} rows; "
          "always check rowcount.")

    print("\n[9] DROP TABLE on a referenced table")
    try:
        db.execute("DROP TABLE teacher")
    except Error as err:
        print(f"  Refused as expected: {err.msg}")
        print("  The course table still references teacher. Drop child tables first,")
        print("  or remove the foreign key, instead of disabling FOREIGN_KEY_CHECKS.")

    print("\n[10] Transaction with rollback")
    db.conn.start_transaction()
    try:
        with closing(db.conn.cursor()) as cur:
            cur.execute("INSERT INTO client VALUES (107, 'Half Done Ltd', 'Somewhere 1', 'Test')")
            # This fails: client 101 already exists (duplicate primary key)
            cur.execute("INSERT INTO client VALUES (101, 'Duplicate', 'x', 'x')")
        db.conn.commit()
    except Error as err:
        db.conn.rollback()
        print(f"  Second insert failed ({err.msg}); the whole transaction was rolled back.")
    exists = db.query("SELECT COUNT(*) AS n FROM client WHERE client_id = 107")["n"][0]
    print(f"  Client 107 present after rollback: {bool(exists)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", default="school_demo")
    parser.add_argument("--reset", action="store_true",
                        help="drop and recreate the demo database first")
    args = parser.parse_args()

    host = os.environ.get("MYSQL_HOST", "localhost")
    user = os.environ.get("MYSQL_USER")
    password = os.environ.get("MYSQL_PASSWORD")
    if not user or password is None:
        raise SystemExit("Set MYSQL_USER and MYSQL_PASSWORD (and optionally MYSQL_HOST).")

    print("[1] Connecting to the server and creating the database")
    server = Database(host, user, password)
    if args.reset:
        server.execute(f"DROP DATABASE IF EXISTS `{args.database}`")
    server.execute(f"CREATE DATABASE IF NOT EXISTS `{args.database}` "
                   "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    server.close()

    db = Database(host, user, password, args.database)
    try:
        build(db)
        read_and_join(db)
        aggregate(db)
        modify(db)
    finally:
        db.close()


if __name__ == "__main__":
    main()
