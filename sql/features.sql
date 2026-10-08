-- One row of behaviour features per customer. Standard SQL: aggregates, CASE and NULLIF only.
WITH per_customer AS (
    SELECT
        s.customer_id,
        MAX(CASE WHEN s.month_index = 1 THEN s.repayment_status END)               AS latest_status,
        MAX(s.repayment_status)                                                   AS worst_status,
        AVG(s.repayment_status)                                                   AS avg_status,
        SUM(CASE WHEN s.repayment_status > 0 THEN 1 ELSE 0 END)                   AS months_delayed,
        SUM(CASE WHEN s.repayment_status > 0 AND s.month_index <= 3 THEN 1 ELSE 0 END) AS months_delayed_last3,
        MAX(CASE WHEN s.month_index = 1 THEN s.bill_amount END)                   AS latest_bill,
        AVG(s.bill_amount)                                                        AS avg_bill,
        MAX(CASE WHEN s.month_index = 1 THEN s.bill_amount END)
          - MAX(CASE WHEN s.month_index = 6 THEN s.bill_amount END)               AS bill_change_6m,
        SUM(s.paid_amount)                                                        AS total_paid,
        MAX(CASE WHEN s.month_index = 1 THEN s.paid_amount END)                   AS latest_paid,
        SUM(CASE WHEN s.paid_amount = 0 AND s.bill_amount > 0 THEN 1 ELSE 0 END)  AS months_unpaid,
        SUM(s.paid_amount) / NULLIF(SUM(CASE WHEN s.bill_amount > 0 THEN s.bill_amount END), 0)
                                                                                  AS paid_to_billed_ratio
    FROM statements AS s
    GROUP BY s.customer_id
)
SELECT
    c.customer_id,
    c.credit_limit,
    p.latest_status,
    p.worst_status,
    p.avg_status,
    p.months_delayed,
    p.months_delayed_last3,
    p.latest_bill,
    p.avg_bill,
    p.bill_change_6m,
    p.latest_bill / NULLIF(c.credit_limit, 0) AS latest_utilisation,
    p.avg_bill    / NULLIF(c.credit_limit, 0) AS avg_utilisation,
    p.total_paid,
    p.latest_paid,
    p.months_unpaid,
    COALESCE(p.paid_to_billed_ratio, 1.0)     AS paid_to_billed_ratio,
    o.paid_late
FROM customers AS c
JOIN per_customer AS p ON p.customer_id = c.customer_id
LEFT JOIN outcomes AS o ON o.customer_id = c.customer_id
ORDER BY c.customer_id;
