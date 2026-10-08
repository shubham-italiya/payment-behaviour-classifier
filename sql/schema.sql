-- Tables built from the raw UCI file (one wide row per customer) by payclf.data.
-- month_index 1 = September 2005 (the most recent month) ... 6 = April 2005.

CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY,
    credit_limit  REAL    NOT NULL,
    -- Demographics are stored for completeness but deliberately NOT used as model features.
    sex           INTEGER,
    education     INTEGER,
    marriage      INTEGER,
    age           INTEGER
);

CREATE TABLE IF NOT EXISTS statements (
    customer_id      INTEGER NOT NULL REFERENCES customers (customer_id),
    month_index      INTEGER NOT NULL CHECK (month_index BETWEEN 1 AND 6),
    bill_amount      REAL    NOT NULL,  -- amount on the statement
    paid_amount      REAL    NOT NULL,  -- amount paid in that month
    repayment_status INTEGER NOT NULL,  -- -2/-1/0 = paid on time or no use; 1..9 = months of delay
    PRIMARY KEY (customer_id, month_index)
);

CREATE TABLE IF NOT EXISTS outcomes (
    customer_id  INTEGER PRIMARY KEY REFERENCES customers (customer_id),
    paid_late    INTEGER NOT NULL CHECK (paid_late IN (0, 1))  -- defaulted on the next month's payment
);
